import random
from pySPICElib.SPICEtools import *
from PSOA.pointres import pointres
from PSOA.radarcover import radarcover
from PSOA.groundtrack import groundtrack
import spiceypy as sp

from FuturePackage import DataManager


class oplan():
    def __init__(self, substart = None, subend = None):
        self.subproblem = [substart, subend]
        roiL1, roiL2 = DataManager.getInstance().getROIList() #è sbagliato perchè non consente di selezionare il subproblem.
        # in ogni caso si dovrebbero definire un substart e subend per ogni strumento (quindi per ogni lista), dato che
        # le liste hanno dimensioni diverse
        self.stol1 = [0.] * len(roiL1)  # start time observation list
        self.qroi1 = [0.] * len(roiL1)  # cache for the fitness of each ROI observation
        self.obsLength1 = [0.] * len(roiL1)  # cache for the duration of the observation for each ROI
        self.stol2 = [0.] * len(roiL2)  # start time observation list
        self.qroi2 = [0.] * len(roiL2)  # cache for the fitness of each ROI observation
        self.obsLength2 = [0.] * len(roiL2)  # cache for the duration of the observation for each ROI

    def removeEmptyTW(self, unconstrainedTW, roiL):
        roiL1, roiL2 = DataManager.getInstance().getROIList()
        tws = []
        if spice.wncard(unconstrainedTW) == 1:
            print("Unconstrained Search Space is a unique TW")
            return unconstrainedTW
        for tw in range(spice.wncard(unconstrainedTW)):
            twbeg, twend = self.getTWBeginEnd(tw)
            isempty = True
            for roi in roiL:
                if not isempty: break
                for interval in range(spice.wncard(roi)):
                    twStart, twEnd = spice.wnfetd(roi, interval)
                    if twStart >= twbeg and twEnd <= twend:
                        isempty = False
                        tws.append(tw)
                        break
        r = stypes.SPICEDOUBLE_CELL(len(tws) * 2)
        for i in range(len(tws)):
            t0, t1 = spice.wnfetd(unconstrainedTW, i)
            spice.wninsd(t0, t1, r)
        return r

    def getNgoals(self):
        return 2
    
    def getTWBeginEnd(self, tw, interval):  # ARREGLAR returns the et begin & end of a TW interval
        TWBegin, TWEnd = spice.wnfetd(tw, interval)
        return TWBegin, TWEnd

    def print_auxdata(self):
        roiL = DataManager.getInstance().getROIList()
        instrument = DataManager.getInstance().getInstrumentData()
        observer = DataManager.getInstance().getObserver()

        print('observer:', observer)
        print('target:', roiL[0].body)
        print('Regions to be studied:', )
        for roi in roiL:
            print('     -', roi.name)
        print('ROIS in this plan ', len(roiL))
        for i, roi in enumerate(roiL):
            print_tw(roi.ROI_TW, '(s) for ' + roi.name)
        print('ifov=', instrument.ifov)

    def getAllTw(self, i):  #   returns a single interval tw plus start, end covering all tw;
        roiL = DataManager.getInstance().getROIList()
        sa = float('inf')  # first start
        ea = -float('inf')  # last end
        TWbeg, TWend = self.getTWBeginEnd(i)
        for tw in roiL:  # NOT LIKE THIS
            for interval in range(spice.wncard(tw)):
                s, e = spice.wnfetd(tw, interval)
                if s >= TWbeg and e <= TWend:  # If the interval of the general timewindow (the one that contains info for all fbs and rois) is within the flyby we are studying
                    if s < sa: sa = s
                    if e > ea: ea = e
                if sa != float('inf') and ea != -float('inf'):
                    r = newTimeWindow(sa, ea)
        return r, sa, ea

    def getObsLength(self, roi, et):
        interval, _, _ = self.findIntervalInTw(et, roi.ROI_TW)
        _, timeobs, _ = roi.interpolateObservationData(et, interval)
        return timeobs

    # evals a metric of Res of all ROIs (the LOWER, the better)
    # in this case, averaged centroid res. in km/pix

    def evalResPlan(self):
        for i in range(len(self.stol1)):
            ts = self.stol1[i]
            te = ts + self.obsLength1[i]
            et = np.linspace(ts, te, 4)
            qv = []
            for t in et:
                qv.append(self.evalResRoi(i, t))
            self.qroi1[i] = sum(qv) / len(et)

        return self.qroi1

    def evalResRoi(self, i, et):  # returns instantaneous resolution (fitness) of roi (integer)
        roiL1, _ = DataManager.getInstance().getROIList()
        observer = DataManager.getInstance().getObserver()
        instrument = DataManager.getInstance().getInstrumentData()
        #print(i)
        _, _, res = roiL1[i].interpolateObservationData(et)
        return res  # pointres(instrument.ifov, roiL[i].centroid, et, roiL[i].body, observer)

    # returns the total overlap time, defined as the sum of the overlaps between consecutive observations
    # overlaps of more that two observations are not considered
    def getTotalOverlapTime(self):
        si = sorted(range(len(self.stol1)), key=lambda k: self.stol1[k])  # ROI sorted by obs time
        sorted_stol = [self.stol1[i] for i in si]
        isfirst = True
        toverlap = 0
        for i in range(len(sorted_stol)):
            if isfirst:
                isfirst = False
                continue
            startt = sorted_stol[i]
            endprevious = sorted_stol[i - 1] + self.obsLength1[si[i - 1]]
            overlap = 0
            if endprevious > startt: overlap = endprevious - startt
            toverlap = toverlap + overlap
        return toverlap
    
    def ranFun(self):
        roiL, _ = DataManager.getInstance().getROIList()
        for i, roi in enumerate(roiL):
            #print(roi.name)
            _, rr, obslen = self.uniformRandomInTw(roi)
            self.stol1[i] = rr
            self.obsLength1[i] = obslen

    def mutFun(self, f=0, g=0):
        roiL, _ = DataManager.getInstance().getROIList()
        for i, roi in enumerate(roiL):
            currentBegin = self.stol1[i]
            newBeginEt, obslen = self.randomSmallChangeIntw(currentBegin, roi, f)
            self.stol1[i] = newBeginEt
            self.obsLength1[i] = obslen

    def fitFun(self):
        tov = self.getTotalOverlapTime()
        if tov > 0:
            return [tov * 1e9, tov / 1e9]
        return [np.mean(self.evalResPlan()),-self.evalCovScan()]


    def uniformRandomInTw(self, roi):
        nint = spice.wncard(roi.ROI_TW)
        plen = [0] * nint
        outOfTW = True

        for i in range(nint):
            intbeg, intend = spice.wnfetd(roi.ROI_TW, i)
            plen[i] = intend - intbeg
        total = sum(plen)
        probabilities = [p / total for p in plen]
        val = list(range(nint))
        # Select one float randomly with probability proportional to its value
        psel = random.choices(val, weights=probabilities)[0]
        i0, i1 = spice.wnfetd(roi.ROI_TW, psel)
        while outOfTW:
            rr = random.uniform(i0, i1)
            #print(f'start obs:{rr}')
            #print(f'end interval:{i1}')
            obslen = self.getObsLength(roi, rr)
            #print(f'length obs {obslen}')
            #obslen = i1 - rr
            if rr + obslen <= i1:
                outOfTW = False
        return psel, rr, obslen

    #     # given time t and a tw with multiple intervals, returns the number of interval contaning t, the start and the end
    #     # returns -1,0.0,0.0 if t is not contained in any interval
    def findIntervalInTw(self, t, tw):
        nint = spice.wncard(tw)
        for i in range(nint):
            intbeg, intend = spice.wnfetd(tw, i)
            if intbeg <= t <= intend:
                return i, intbeg, intend
        return -1, 0.0, 0.0

    # given time t0, with both t0 and t0+olen belonging to the same interval in time window tw,
    # returns another instant t0new so that t0new and t0new+olen are in the same tw interval as before
    # t0new-t0 distributes N(0,sigma), when the conditions are satisfied

    def randomSmallChangeIntw(self, t0, roi, f):
        i, intervals, intervalend = self.findIntervalInTw(t0, roi.ROI_TW)
        if np.abs(t0 - intervals) >= np.abs(t0 - intervalend):
            sigma0 = np.abs(intervalend - t0)
        else:
            sigma0 = np.abs(intervals - t0)
        sigma = sigma0
        ns = 0
        while True:
            newBegin = t0 + np.random.normal(0, sigma)
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
                #print(t0)
                #print(intervals)
                #print(intervalend)
                raise Exception('uhhh cant find mutation')

        return newBegin, obslen

    def distance(self, other):
        dd = 0
        for i in range(len(self.stol1)):
            q = math.fabs(other.stol1[i] - self.stol1[i])
            if q > dd: dd = q
        return q


    def repFun(self, p1, f1, f2):
        roiL, _ = DataManager.getInstance().getROIList()
        newind = []
        newobs = []
        for i in range(len(p1.stol1)):
            op = (p1.stol1[i] + self.stol1[i])/2
            #print(len(p1.stol))
            #print(len(self.stol))
            a,_,_ = self.findIntervalInTw(op, roiL[i].ROI_TW)
            if a != -1:
                #print('PARENT FOUND')
                newind.append(op)
                obslen = self.getObsLength(roiL[i], op)
                newobs.append(obslen)
            else:
                #print('Op not found, mutating')
                _, rr, obslen = self.uniformRandomInTw(roiL[i])
                newind.append(rr)
                newobs.append(obslen)
        self.stol1 = newind
        self.obsLength1 = newobs

    def computeScanWindow(self):
        JANUS_TW = stypes.SPICEDOUBLE_CELL(2000)
        SCAN_TW = stypes.SPICEDOUBLE_CELL(2000)
        roiL, roiL2 = DataManager.getInstance().getROIList()
        for i in range(len(roiL)):
            tend = self.stol1[i] + self.obsLength1[i]
            sp.wninsd(self.stol1[i], tend, JANUS_TW)
        
        for antij_roi in roiL2:
            SCAN_TW = sp.wnunid(SCAN_TW, antij_roi.ROI_TW)
        radar_TW = sp.wndifd(SCAN_TW, JANUS_TW) # sp.wndifd computes the difference between two time windows (complementary)
        n = sp.wncard(radar_TW)
        #print(n)
        return radar_TW

    def evalCovScan(self):
        target = 'CALLISTO'
        radii = spice.bodvrd(target, "RADII", 3)[1][1] # [km]
        scan_tw = self.computeScanWindow()
        observer = DataManager.getInstance().getObserver()
        n = sp.wncard(scan_tw)
        #print(n)
        interval_cov = []
        for i in range(n):
            tstart, tend = sp.wnfetd(scan_tw, i)
            #print(f'{i}: start {tstart} and end {tend}')
            et = np.linspace(tstart, tend, 4)
            qv = []
            for t in et:
                qv.append(radarcover(radii = radii, srfpoint= groundtrack(observer, t, target), t = t, target = target, obs = observer))
            interval_cov.append(sum(qv) / len(et))
        #print(sum(interval_cov))
        return sum(interval_cov)
    
    def getNImages(self, instrument, observer):
        roiL, _ = DataManager.getInstance().getROIList()
        numimg = 0
        timeobservation = 0
        for i, roi in enumerate(roiL):
            interval, _, _ = self.findIntervalInTw(self.stol1[i], roi.ROI_TW)
            nImages, timeobs, res = roi.interpolateObservationData(self.stol1[i], interval)
            numimg = numimg + nImages
            timeobservation = timeobservation + timeobs
            #roi_TW = stypes.SPICEDOUBLE_CELL(2000)
            #tend = self.stol1[i] + self.obsLength1[i]
            #sp.wninsd(self.stol1[i], tend, roi_TW)
            #roi.initializeObservationDataBase(roi_TW, instrument, observer)
            #nImages.append([roi.name, np.mean(roi.ROI_ObsRes), np.sum(roi.ROI_ObsImg)])

        return [numimg, timeobservation, res]

    def plotObservations(self, ax, fig):
        intervals_janus = []
        intervals_rime = []
        roiL, _ = DataManager.getInstance().getROIList()
        for i, roi in enumerate(roiL):
            tend = self.stol1[i] + self.obsLength1[i]
            intervals_janus.append([roi.name, roi.vertices, self.stol1[i], tend])
        scan_tw = self.computeScanWindow()
        nint = spice.wncard(scan_tw)
        for i in range(nint):
            intbeg, intend = spice.wnfetd(scan_tw, i)
            intervals_rime.append([intbeg, intend])
        rois_names = ['JANUS_CAL_4_3_03', 'JANUS_CAL_4_3_10', 'JANUS_CAL_4_7_06', 'JANUS_CAL_6_1_07', 'JANUS_CAL_6_1_08']
        #colors = [cmap(i) for i in np.linspace(0, 1, len(intervals_janus) + 2)]
        colors = ['purple', 'white', 'green', 'tab:olive', 'pink']
        #gtlon, gtlat = groundtrack('JUICE', et, roiL[0].body)
        tstep = 0.5
        for i, interval_j in enumerate(intervals_janus):
            et = np.arange(interval_j[2], interval_j[3] + tstep, tstep)
            gtlon, gtlat = groundtrack('JUICE', et, roiL[0].body)
            for j in range(len(et)):
                ax.scatter(gtlon[j], gtlat[j], c=colors[i], s=3, marker= 's')
            #found = False
            #for j, t in enumerate(et):
            #    if t > interval_j[2] and t <= interval_j[3]:
            #        found = True
            #        ax.scatter(gtlon[j], gtlat[j], c=colors[i], s=3)
            
            #if found:
            vert_x = [0]*(len(interval_j[1])+1)
            vert_y = [0]*(len(interval_j[1])+1)
            for k in range(len(interval_j[1])):
                if interval_j[1][k][0] < 0:
                    vert_x[k] = interval_j[1][k][0] + 180  
                elif interval_j[1][k][0] > 0:
                     vert_x[k] = interval_j[1][k][0] - 180
                vert_y[k] =  interval_j[1][k][1]
            vert_x[-1] = vert_x[0]
            vert_y[-1] = vert_y[0]
            ax.plot(np.array(vert_x), np.array(vert_y), c = colors[i], linewidth = 2, linestyle = '--', label = rois_names[i])

        scatter_lons = []
        scatter_lats = []
        scatter_alts = []
        for i, interval_r in enumerate(intervals_rime):
            et = np.arange(interval_r[0], interval_r[1] + tstep, tstep)
            for t in et:
                alt = altitude('JUICE', roiL[0].body, t)
                gtlon, gtlat = groundtrack('JUICE', t, roiL[0].body)
                scatter_lons.append(gtlon)
                scatter_lats.append(gtlat)
                scatter_alts.append(alt)
            scatter = ax.scatter(scatter_lons, scatter_lats, marker= 's' , c='r', s=3)
            
            #for j, t in enumerate(et):
            #    if t > interval_r[0] and t <= interval_r[1]:
            #        ax.scatter(gtlon[j], gtlat[j], c=colors[-2], s=3)
        coolwarm_color = plt.cm.coolwarm(0.75)
        ax.plot([], [], c='r', linewidth = 2, linestyle = '-', label = 'RIME')
        #cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', extend='max', pad=0.1, shrink=0.6)
        #cbar.set_label('RIME Altitude (km)', fontsize=15)
        legend = ax.legend(loc='upper center', fontsize=12)
        legend.get_frame().set_facecolor('none')
        for text in legend.get_texts():
            text.set_color("white")
        #ax.set_title("Ground track", fontsize=18)
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        #ax.grid(True, which='minor')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(r'Longitude (°)', fontsize=18)
        ax.set_ylabel(r'Latitude (°)', fontsize=18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)


    def plotObservations_2(self, ax, fig):
        intervals_janus = []
        #intervals_rime = []
        roiL, _ = DataManager.getInstance().getROIList()
        for i, roi in enumerate(roiL):
            tend = self.stol1[i] + self.obsLength1[i]
            intervals_janus.append([roi.name, roi.vertices, self.stol1[i], tend])
        #scan_tw = self.computeScanWindow()
        #nint = spice.wncard(scan_tw)
        #for i in range(nint):
        #    intbeg, intend = spice.wnfetd(scan_tw, i)
        #    intervals_rime.append([intbeg, intend])
        rois_names = ['JANUS_CAL_4_3_03', 'JANUS_CAL_4_3_10', 'JANUS_CAL_4_7_06', 'JANUS_CAL_6_1_07', 'JANUS_CAL_6_1_08']

        #cmap = plt.get_cmap('rainbow')
        #colors = [cmap(i) for i in np.linspace(0, 1, len(intervals_janus) + 2)]
        colors = ['purple', 'white', 'green', 'tab:olive', 'pink']
        #gtlon, gtlat = groundtrack('JUICE', et, roiL[0].body)
        tstep = 0.5
        for i, interval_j in enumerate(intervals_janus):
            et = np.arange(interval_j[2], interval_j[3] + tstep, tstep)
            gtlon, gtlat = groundtrack('JUICE', et, roiL[0].body)
            for j in range(len(et)):
                ax.scatter(gtlon[j], gtlat[j], c=colors[i], s=3, marker= 's')
            #found = False
            #for j, t in enumerate(et):
            #    if t > interval_j[2] and t <= interval_j[3]:
            #        found = True
            #        ax.scatter(gtlon[j], gtlat[j], c=colors[i], s=3)
            
            #if found:
            vert_x = [0]*(len(interval_j[1])+1)
            vert_y = [0]*(len(interval_j[1])+1)
            for k in range(len(interval_j[1])):
                if interval_j[1][k][0] < 0:
                    vert_x[k] = interval_j[1][k][0] + 180  
                elif interval_j[1][k][0] > 0:
                     vert_x[k] = interval_j[1][k][0] - 180
                vert_y[k] =  interval_j[1][k][1]
            vert_x[-1] = vert_x[0]
            vert_y[-1] = vert_y[0]
            ax.plot(np.array(vert_x), np.array(vert_y), c = colors[i], linewidth = 2, linestyle = '--', label = rois_names[i])

        #scatter_lons = []
        #scatter_lats = []
        #scatter_alts = []
        #for i, interval_r in enumerate(intervals_rime):
        #    et = np.arange(interval_r[0], interval_r[1] + tstep, tstep)
        #    for t in et:
        #        alt = altitude('JUICE', roiL[0].body, t)
        #        gtlon, gtlat = groundtrack('JUICE', t, roiL[0].body)
        #        scatter_lons.append(gtlon)
        #        scatter_lats.append(gtlat)
        #        scatter_alts.append(alt)
        #    scatter = ax.scatter(scatter_lons, scatter_lats, marker= 's' , c=scatter_alts, cmap='coolwarm_r', s=3)
            
            #for j, t in enumerate(et):
            #    if t > interval_r[0] and t <= interval_r[1]:
            #        ax.scatter(gtlon[j], gtlat[j], c=colors[-2], s=3)
        #coolwarm_color = plt.cm.coolwarm(0.75)
        #ax.plot([], [], c=coolwarm_color, linewidth = 2, linestyle = '-', label = 'RIME')
        #cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', extend='max', pad=0.1, shrink=0.6)
        #cbar.set_label('RIME Altitude (km)', fontsize=15)
        legend = ax.legend(loc='upper center', fontsize=12)
        legend.get_frame().set_facecolor('none')
        for text in legend.get_texts():
            text.set_color("white")
        #ax.set_title("Ground track", fontsize=18)
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        #ax.grid(True, which='minor')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(r'Longitude (°)', fontsize=18)
        ax.set_ylabel(r'Latitude (°)', fontsize=18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)



    def plotObservations_3(self, ax, fig):
        #intervals_janus = []
        intervals_rime = []
        roiL, _ = DataManager.getInstance().getROIList()
        #for i, roi in enumerate(roiL):
        #    tend = self.stol1[i] + self.obsLength1[i]
        #    intervals_janus.append([roi.name, roi.vertices, self.stol1[i], tend])
        
        scan_tw = self.computeScanWindow()
        nint = spice.wncard(scan_tw)
        for i in range(nint):
            intbeg, intend = spice.wnfetd(scan_tw, i)
            intervals_rime.append([intbeg, intend])


        #cmap = plt.get_cmap('rainbow')
        ##colors = [cmap(i) for i in np.linspace(0, 1, len(intervals_janus) + 2)]
        #colors = ['purple', 'white', 'green', 'tab:olive', 'pink']
        ##gtlon, gtlat = groundtrack('JUICE', et, roiL[0].body)
        tstep = 0.5
        #for i, interval_j in enumerate(intervals_janus):
        #    et = np.arange(interval_j[2], interval_j[3] + tstep, tstep)
        #    gtlon, gtlat = groundtrack('JUICE', et, roiL[0].body)
        #    for j in range(len(et)):
        #        ax.scatter(gtlon[j], gtlat[j], c=colors[i], s=3, marker= 's')
        #    #found = False
        #    #for j, t in enumerate(et):
        #    #    if t > interval_j[2] and t <= interval_j[3]:
        #    #        found = True
        #    #        ax.scatter(gtlon[j], gtlat[j], c=colors[i], s=3)
        #    
        #    #if found:
        #    vert_x = [0]*(len(interval_j[1])+1)
        #    vert_y = [0]*(len(interval_j[1])+1)
        #    for k in range(len(interval_j[1])):
        #        if interval_j[1][k][0] < 0:
        #            vert_x[k] = interval_j[1][k][0] + 180  
        #        elif interval_j[1][k][0] > 0:
        #             vert_x[k] = interval_j[1][k][0] - 180
        #        vert_y[k] =  interval_j[1][k][1]
        #    vert_x[-1] = vert_x[0]
        #    vert_y[-1] = vert_y[0]
        #    ax.plot(np.array(vert_x), np.array(vert_y), c = colors[i], linewidth = 2, linestyle = '--', label = interval_j[0])

        scatter_lons = []
        scatter_lats = []
        scatter_alts = []
        for i, interval_r in enumerate(intervals_rime):
            et = np.arange(interval_r[0], interval_r[1] + tstep, tstep)
            for t in et:
                alt = altitude('JUICE', roiL[0].body, t)
                gtlon, gtlat = groundtrack('JUICE', t, roiL[0].body)
                scatter_lons.append(gtlon)
                scatter_lats.append(gtlat)
                scatter_alts.append(alt)
            scatter = ax.scatter(scatter_lons, scatter_lats, marker= 's' , c=scatter_alts, cmap = 'hot',  s=3)
            
            #for j, t in enumerate(et):
            #    if t > interval_r[0] and t <= interval_r[1]:
            #        ax.scatter(gtlon[j], gtlat[j], c=colors[-2], s=3)
        #ax.plot([], [], c='r', linewidth = 2, linestyle = '-', label = 'RIME')
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', extend='max', pad=0.12, shrink=0.6)
        cbar.set_label('RIME Altitude (km)', fontsize=15)
        #legend = ax.legend(loc='upper center', fontsize=12)
        #legend.get_frame().set_facecolor('none')
        #for text in legend.get_texts():
        #    text.set_color("white")
        #ax.set_title("Ground track", fontsize=18)
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        #ax.grid(True, which='minor')
        ax.set_aspect('equal', 'box')
        ax.set_xlabel(r'Longitude (°)', fontsize=18)
        ax.set_ylabel(r'Latitude (°)', fontsize=18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)






    def getNdof(self):
        return len(self.stol1)

    def getVector(self):
        return self.obsLength1, self.stol1, self.qroi1, self.obsLength2, self.stol2, self.qroi2

    def replaceWithVector(self, newvect):
        self.stol1 = newvect
