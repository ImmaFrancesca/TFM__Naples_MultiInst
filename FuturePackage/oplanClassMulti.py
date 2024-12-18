import copy
import math
import numpy as np
import spiceypy as spice
from spiceypy.utils.support_types import SPICEDOUBLE_CELL
import random
import plotly.graph_objects as go
import warnings
from FuturePackage import DataManager
from PSOA import *

class oplan():
    def __init__(self, n, substart = None, subend = None):
        self.Ninst = n # number of instruments
        self.subproblem = [substart, subend]
        roisL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1]) # list of list each made of oROI instances, contains the ROIs of each instrument
        self.stol = [np.zeros(len(roiL)) for roiL in roisL] # list of arrays, each contains the starting instants of
        #observation for each ROI of the corresponding instrument
        self.obsLen = [np.zeros(len(roiL)) for roiL in roisL] # list of arrays, each contains the  length of the
        #observation for each ROI of the corresponding instrument
        self.qroi = [np.zeros(len(roiL)) for roiL in roisL] # list of arrays, each contains the resolution of the
        #observation for each ROI of the corresponding instrument
        self.croi = [np.zeros(len(roiL)) for roiL in roisL] # list of arrays, each contains the coverage of the
        #observation for each ROI of the corresponding instrument (if this is a camera,otherwise it contains an array of 0)

    def getNgoals(self):
        return  self.Ninst # PROBLEM_CHANGE :could be the number of instruments, if for every instrument we have an objective function

    def getObsLength(self, roi, et):
        interval, _, _ = self.findIntervalInTw(et, roi.ROI_TW)
        if roi.mosaic:
            _, timeobs, _, _ = roi.interpolateObservationData(et, interval)
        else:
            _, timeobs, _ = roi.interpolateObservationData(et, interval)
        return timeobs

    def getNImages(self, inst_index, observer):
        roiL = DataManager.getInstance().getROIList()[inst_index]
        numimg = 0
        timeobservation = 0
        for i, roi in enumerate(roiL):
            interval, _, _ = self.findIntervalInTw(self.stol[inst_index][i], roi.ROI_TW)
            if roi.mosaic:
                nImages, timeobs, res, _ = roi.interpolateObservationData(self.stol[inst_index][i], interval)
            else:
                nImages, timeobs, res = roi.interpolateObservationData(self.stol[inst_index][i], interval)
            numimg = numimg + nImages
            #timeobservation = timeobservation + timeobs
            #roi_TW = stypes.SPICEDOUBLE_CELL(2000)
            #tend = self.stol1[i] + self.obsLength1[i]
            #sp.wninsd(self.stol1[i], tend, roi_TW)
            #roi.initializeObservationDataBase(roi_TW, instrument, observer)
            #nImages.append([roi.name, np.mean(roi.ROI_ObsRes), np.sum(roi.ROI_ObsImg)])
        #return [numimg, timeobservation, res]
        return numimg

    def ranFun(self):
        n_trials = 0 # number of trials to search for a feasible individual
        feasible = False
        while n_trials < 50 and not feasible:
            n_trials += 1
            print('n_trials es',n_trials)
            roisL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1])
            ran_instL = random.sample(list(range(len(roisL))), len(roisL))
            ran_roiL = []
            for i in ran_instL:
                ran_indList = random.sample(list(range(len(roisL[i]))), len(roisL[i]))
                ran_roiL.append(ran_indList)
            assigned_tws = SPICEDOUBLE_CELL(2000)

            for k, i in enumerate(ran_instL):
                for j in ran_roiL[k]:
                    roi = roisL[i][j]
                    current_tw = roi.ROI_TW
                    forbidden_tw = spice.wnintd(current_tw, assigned_tws)
                    new_tw = spice.wndifd(current_tw, forbidden_tw)
                    feasible = self.checkROI(roi, new_tw)
                    if feasible:
                        flag = True
                        nit = 0
                        while flag and nit < 20:
                            _, rr, obsLength, flag = self.uniformRandomInTw(new_tw, roi)
                            nit += 1
                        if flag:
                            # print('ranFun: Random individual not found for this order of instruments and ROIs.')
                            feasible = False
                            break
                        self.stol[i][j] = rr
                        self.obsLen[i][j] = obsLength
                        spice.wninsd(rr, rr + obsLength, assigned_tws)
                    else:
                        break
                if not feasible:
                    break
        if not feasible:
            raise Exception('It is not possible to find a random individual')


    def uniformRandomInTw(self, tw, roi):
        nint = spice.wncard(tw)
        plen = [0] * nint
        outOfTW = True

        for i in range(nint):
            intbeg, intend = spice.wnfetd(tw, i)
            plen[i] = intend - intbeg
        total = sum(plen)
        probabilities = [p / total for p in plen]
        val = list(range(nint))
        # Select one float randomly with probability proportional to its value

        while outOfTW is True:
            if val == []:
                break # in this case, r and obslen are the last obtained. There will be overlap with the observation of
                # the other rois or the observation falls out of the compilant TW (initial one)

            psel = random.choices(val, weights = probabilities)[0]
            index = val.index(psel)
            val.remove(psel)
            probabilities.remove(probabilities[index])

            i0, i1 = spice.wnfetd(tw, psel)
            n_it = 0
            while outOfTW and n_it < 50:
                n_it += 1
                rr = random.uniform(i0, i1)
                obslen = self.getObsLength(roi, rr)
                if rr + obslen <= i1:
                    outOfTW = False
        return psel, rr, obslen, outOfTW

    def mutFun(self, f = 0, g = 0):
        roisL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1])
        ran_instL = random.sample(list(range(len(roisL))), len(roisL))
        ran_roiL = []
        for i in ran_instL:
            ran_indList = random.sample(list(range(len(roisL[i]))), len(roisL[i]))
            ran_roiL.append(ran_indList)

        n_inst = 0
        for k, i in enumerate(ran_instL):
            otherROIStw = SPICEDOUBLE_CELL(2000)
            for ind in range(self.Ninst):
                if ind != i:
                    for ind_ in range(len(self.stol[ind])):
                        start = self.stol[ind][ind_]
                        end = start + self.obsLen[ind][ind_]
                        spice.wninsd(start, end, otherROIStw)
            stol = copy.deepcopy(self.stol[i])
            obsLen = copy.deepcopy(self.obsLen[i])
            for roi_index, j in enumerate(ran_roiL[k]):
                roi = roisL[i][j]
                current_tw = roi.ROI_TW
                forbidden_tw = spice.wnintd(current_tw, otherROIStw)
                new_tw = spice.wndifd(current_tw, forbidden_tw)
                newStart, newLength, flag = self.randomSmallChangeIntw(self.stol[i][j], roi, new_tw, f)
                if not flag:
                    self.stol[i][j] = newStart
                    self.obsLen[i][j] = newLength
                elif roi_index != 0:
                    self.stol[i] = copy.deepcopy(stol)
                    self.obsLen[i] = copy.deepcopy(obsLen)
                    n_inst += 1
                    break
                spice.wninsd(self.stol[i][j], self.stol[i][j] + self.obsLen[i][j], otherROIStw)
        if n_inst == self.Ninst:
            print('Mutation has not been performed on the current sample.')

    def randomSmallChangeIntw(self, t0, roi, tw, f):
        flag = False
        i, intervals, intervalend = self.findIntervalInTw(t0, tw)
        if i == -1: # a new random initial instant is chosen within tw

            feasibility = self.checkROI(roi, tw)
            if not feasibility:
                return None, None, True

            flag_ = True
            nit = 0
            #print('Interval not found, a new random initial instant is chosen')
            while flag_ and nit < 20:
                _, newBegin, obsLen, flag_ = self.uniformRandomInTw(tw, roi)
                nit += 1
            if flag_:
            #    print('New random initial instant not found.')
                flag = True
            return newBegin, obsLen, flag

        interval = SPICEDOUBLE_CELL(2000)
        spice.wninsd(intervals, intervalend, interval)
        feasibility = self.checkROI(roi, interval)
        if not feasibility:
            return None, None, True
        if np.abs(t0 - intervals) >= np.abs(t0 - intervalend):
            sigma0 = np.abs(intervalend - t0)
        else:
            sigma0 = np.abs(intervals - t0)
        sigma = sigma0
        ns = 0
        while True:
            newBegin = t0 + np.random.normal(0, sigma)
            if newBegin < 0:
                #print('newBegin = ', newBegin)
                newBegin = 0
            if newBegin > 20:
                #print('newBegin = ', newBegin)
                newBegin = 20
            obslen = self.getObsLength(roi, newBegin)
            newEnd = newBegin + obslen
            # print('NewEnd ', newEnd)
            # print('Intervalend', intervalend)
            if newBegin >= intervals and newEnd <= intervalend:
                # print('Mutation with sigma = ' + str(sigma) + 's')
                break
            ns += 1
            # print('iteration ', ns)
            if ns > 50:
                # print('halving')
                sigma0 = sigma0 / 2
                sigma = sigma0
            if ns > 500:
                print(t0)
                print(intervals)
                print(intervalend)
                #raise Exception('uhhh cant find mutation')
                flag = True
                #warnings.warn('Cannot find the mutation for the current ROI.')
                break
        return newBegin, obslen, flag

    def findIntervalInTw(self, t, tw):
        nint = spice.wncard(tw)
        for i in range(nint):
            intbeg, intend = spice.wnfetd(tw, i)
            if intbeg <= t <= intend:
                return i, intbeg, intend
        return -1, 0.0, 0.0

    def repFun(self, p1, f1, f2):
        roisL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1])
        feasible = False
        n_rep = 0
        while n_rep < 20 and not feasible:
            n_rep += 1
            print('n_rep =',n_rep)
            ran_instL = random.sample(list(range(len(roisL))), len(roisL))
            ran_roiL = []
            for i in ran_instL:
                ran_indList = random.sample(list(range(len(roisL[i]))), len(roisL[i]))
                ran_roiL.append(ran_indList)

            assigned_tws = SPICEDOUBLE_CELL(2000)

            for k, i in enumerate(ran_instL):
                for j in ran_roiL[k]:
                    roi = roisL[i][j]
                    current_tw = roi.ROI_TW
                    forbidden_tw = spice.wnintd(current_tw, assigned_tws)
                    new_tw = spice.wndifd(current_tw, forbidden_tw)
                    start = (self.stol[i][j] + p1.stol[i][j])/2
                    a, _, iend = self.findIntervalInTw(start, new_tw)
                    if a != -1:
                        obslen = self.getObsLength(roi, start)
                        if start + obslen <= iend:
                            feasible = True
                            self.stol[i][j] = start
                            self.obsLen[i][j] = obslen
                            spice.wninsd(start, start + obslen, assigned_tws)
                            continue
                    #if a == -1:
                    #   print('repFun: Interval not found, a new random initial instant is chosen')
                    #else:
                    #    print('repFun: Observation length is such that the end is outside the interval. A new random initial instant is chosen.')
                    feasible = self.checkROI(roi, new_tw)
                    if feasible:
                        flag = True
                        nit = 0

                        while flag and nit < 20:
                            _, rr, obsLength, flag = self.uniformRandomInTw(new_tw, roi)
                            nit += 1
                        if flag:
                            feasible = False
                            print('repFun: Random initial instant not found for the given ROI, with this order of instruments and ROIs.')
                            break
                        self.stol[i][j] = rr
                        self.obsLen[i][j] = obsLength
                        spice.wninsd(rr, rr + obsLength, assigned_tws)
                    else:
                        break
                if not feasible:
                    break
        if not feasible:
            warnings.warn('Cannot find child individual. Parent 1 is kept.')
            # raise Exception('Cannot find child individual')

    def evalResPlan(self, index_inst):
        roiL = DataManager.getInstance().getROIList()[index_inst]
        if roiL[0].mosaic:
            for i in range(len(self.stol[index_inst])):
                self.qroi[index_inst][i] = self.evalResRoi(index_inst, i, self.stol[index_inst][i])
        else:
            for i in range(len(self.stol[index_inst])):
                ts = self.stol[index_inst][i]
                te = ts + self.obsLen[index_inst][i]
                et = np.linspace(ts, te, 4)
                qv = []
                for t in et:
                    qv.append(self.evalResRoi(index_inst, i, t))
                self.qroi[index_inst][i] = sum(qv) / len(et)
        return self.qroi[index_inst]

    def evalResRoi(self, index_inst, index_roi, et):  # returns instantaneous resolution (fitness) of roi (integer)
        roiL = DataManager.getInstance().getROIList()[index_inst]
        observer = DataManager.getInstance().getObserver()
        instrument = DataManager.getInstance().getInstrumentData()[index_inst]
        #print(i)
        if roiL[index_roi].mosaic:
            _, _, res, _ = roiL[index_roi].interpolateObservationData(et)
        else:
            _, _, res = roiL[index_roi].interpolateObservationData(et)

        return res  # pointres(instrument.ifov, roiL[i].centroid, et, roiL[i].body, observer)

    def evalCovPlan(self, index_inst):
        for i in range(len(self.stol[index_inst])):
            self.croi[index_inst][i] = self.evalCovRoi(index_inst, i, self.stol[index_inst][i])
        return self.croi[index_inst]

    def evalCovRoi(self, index_inst, index_roi, et):  # returns instantaneous resolution (fitness) of roi (integer)
        roiL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1])[index_inst]
        _, _, _, cov = roiL[index_roi].interpolateObservationData(et)
        return cov

    def fitFun(self): # the fitness is the mean value of the two functions to be minimized
        instruments = DataManager.getInstance().getInstrumentData()
        tov = self.getTotalOverlapTime()
        tout = self.getTotalOutOfTWTime()
        if tout != 0:
            print('out of TW')
        if tov != 0:
            print('overlap')
        if tov + tout > 0:
            return [(tov + tout) * 1e9, (tov + tout) * 1e9]
        fitness = []
        for i, instrument in enumerate(instruments):
            if instrument.type == 'CAMERA':
                weights = [0.5, 0.5] # weights needed to evaluate the fitness of EACH camera
                fitness.append(weights[0] * np.mean(self.evalResPlan(i))
                             + weights[1] * np.mean(np.ones(len(self.stol[i])) * 100 - self.evalCovPlan(i)))
            #if instrument.type == 'RADAR':
            #    fitness.append( -self.evalCovScan())
        return fitness

    def getTotalOutOfTWTime(self):
        roisL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1])
        totalout = 0
        for i, instrument in enumerate(roisL):
            for j, roi in enumerate(instrument):
                start = self.stol[i][j]
                end = start + self.obsLen[i][j]
                _, _, intervalend = self.findIntervalInTw(start, roi.ROI_TW)
                if end > intervalend:
                    totalout += end - intervalend
        return totalout

    def getTotalOverlapTime(self):
        all_stol = [roistol for instrument in self.stol for roistol in instrument]
        all_obsLen = [roiobsLen for instrument in self.obsLen for roiobsLen in instrument]
        si = sorted(range(len(all_stol)), key=lambda k: all_stol[k])  # ROI sorted by obs time
        sorted_stol = [all_stol[i] for i in si]
        isfirst = True
        toverlap = 0

        for i in range(len(sorted_stol)):
            if isfirst:
                isfirst = False
                continue
            startt = sorted_stol[i]
            endprevious = sorted_stol[i - 1] + all_obsLen[si[i - 1]]
            overlap = 0
            if endprevious > startt: overlap = endprevious - startt
            toverlap = toverlap + overlap
        return toverlap

    def distance(self, other):
        dd = 0
        for j in range(self.Ninst):
            for i in range(len(self.stol[j])):
                q = math.fabs(other.stol[j][i] - self.stol[j][i])
                if q > dd: dd = q
        return q

    def checkROI(self, roi, tw):
        nint = spice.wncard(tw)

        for i in range(nint):
            intbeg, intend = spice.wnfetd(tw, i)
            interval = np.linspace(intbeg, intend, num = 1000)
            for inst in interval:
                if inst + self.getObsLength(roi, inst) <= intend:
                    return True

        return False


    def checkFeasibility(self):
        roisL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1])

        for i, instrument in enumerate(roisL):
            for roi in instrument:
                tw = roi.ROI_TW
                flag = self.checkROI(roi, tw)
                if not flag:
                    print('The schedule is not feasible since the observation of the', roi.name, 'of the instrument',
                          roi.ROI_InsType, 'is too long')
                    return False
        return True
    def plotGantt(self):
        roisL = DataManager.getInstance().getROIList(self.subproblem[0], self.subproblem[1])
        fig = go.Figure()
        for i in range(self.Ninst):
            for j in range(len(roisL[i])):
                fig.add_shape(
                    type="rect",
                    x0=self.stol[i][j], y0=i - 0.4, x1=self.stol[i][j] + self.obsLen[i][j], y1=i + 0.4,
                    line=dict(color="black", width=1),
                    fillcolor="blue"
                )
                fig.add_annotation(
                    x=(self.stol[i][j] + self.stol[i][j] + self.obsLen[i][j]) / 2,
                    y=i,
                    text=roisL[i][j].name,
                    showarrow=False,
                    yshift=10
                )

        fig.add_shape(type="line",
                      x0=0, x1=0, y0=-1, y1=4,
                      line=dict(color="red", width=2, dash="dash"))
        fig.add_shape(type="line",
                      x0=20, x1=20, y0=-1, y1=4,
                      line=dict(color="red", width=2, dash="dash"))

        fig.update_layout(
            xaxis=dict(
                range=[-0.5, 25],
                title="Time"
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(self.Ninst)),
                ticktext=[i.type for i in DataManager.getInstance().getInstrumentData],
                title="Instrument"
            ),
            shapes=[],
            hovermode='closest'
        )
        fig.show()

    def getVector(self):
        return self.obsLen, self.stol, self.qroi






