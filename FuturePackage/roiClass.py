import math
import numpy as np

import PSOA as psoa
import spiceypy as spice
from pySPICElib.roiDatabase import roi
from area_coverage_planning_python.mosaic_algorithms.paper.precomputation_JUICE.main_onlineFrontier_precompute import mosaicOnlineFrontier

class oPlanRoi(roi):
    def __init__(self, body, name, vertices):
        super().__init__(body, name, vertices)
        self.ROI_InsType = None
        self.ROI_TW = None #[[10, 30], [50, 120]]
        self.ROI_ObsET = None
        self.ROI_ObsLen = None
        self.ROI_ObsImg = None
        self.ROI_ObsRes = None
        self.ROI_ObsCov = None


    def initializeObservationDataBase(self, roitw, instrument=None, observer= None, timeData = None, nImg = None, res = None, mosaic = False):
        self.ROI_TW = roitw  # Compliant TW for a ROI within the mission TW, given certain constraints
        self.ROI_ObsET = self.computeObservationET()
        if timeData is None and nImg is None and res is None:
            timeData, nImg, res = self.computeObservationData(instrument, observer, mosaic)
        self.ROI_ObsLen = timeData
        self.ROI_ObsImg = nImg
        self.ROI_ObsRes = res

    def initializeScanDataBase(self, roitw, instrument=None, observer=None, timeData=None, cov=None, targetRadii = 2634):
        # Ganymede's radius [km] by default
        self.ROI_TW = roitw  # Compliant TW for a ROI within the mission TW, given certain constraints
        self.ROI_ObsET = self.computeObservationET()
        # print(self.ROI_ObsET)
        if timeData is None and cov is None:
            timeData, cov = self.computeScanData(observer=observer, targetRadii=targetRadii)
        self.ROI_ObsLen = timeData
        self.ROI_ObsCov = cov

    def computeObservationET(self):
        et_list = []
        compliantIntervals = spice.wncard(self.ROI_TW)
        for i in range(compliantIntervals):
            twBegin, twEnd = spice.wnfetd(self.ROI_TW, i)
            t = np.linspace(twBegin, twEnd, num=1000, endpoint=True)
            et_list.append(t)
        return et_list

    def computeObservationData(self, instrument, observer, mosaic = False):
        tw_ObsLengths = []
        tw_NImgs = []
        tw_res = []
        for int, compliantInterval in enumerate(self.ROI_ObsET):
            nimg = []
            time = []
            res = []
            if mosaic:
                time, nimg, res =  mosaicOnlineFrontier(compliantInterval, 'JUICE_JANUS', observer, self, instrument, int + 1)
            else:
                for i, et in enumerate(compliantInterval):
                    r = psoa.pointres(instrument.ifov, self.centroid, et, self.body, observer)  # km/pix
                    if np.isnan(r):
                        return np.nan, np.nan
                    areaCov = (r * instrument.npix) ** 2
                    nimg.append(math.ceil((self.area / areaCov) * (1 + instrument.safetyFactor / 100)))
                    time.append(nimg[i] * instrument.imageRate)
                    res.append(r)
            tw_ObsLengths.append(np.array(time))
            tw_NImgs.append(np.array(nimg))
            tw_res.append(np.array(res))
        return tw_ObsLengths, tw_NImgs, tw_res

    def computeScanData(self, observer, targetRadii = 2634):
        tw_ObsLengths = []
        tw_cov = []
        for compliantInterval in self.ROI_ObsET:
            cov = []
            # Calculate the duration of the interval
            start = compliantInterval[0]  # First ET in the interval
            end = compliantInterval[-1]  # Last ET in the interval
            time = end - start  # Duration of the interval

            for i, et in enumerate(compliantInterval):
                # print(len(compliantInterval))
                c = psoa.radarcover(targetRadii, psoa.groundtrack(observer, et, self.body), et, self.body, observer)
                if np.isnan(c):
                    cov.append(np.nan)
                else:
                    cov.append(c)
            tw_ObsLengths.append(np.array(time))
            tw_cov.append(np.array(cov))
        return tw_ObsLengths, tw_cov

    def interpolateObservationData(self, t, interval=None):
        #print(self.name)
        if interval is None:
            for interval in range(len(self.ROI_ObsET)):
                start = self.ROI_ObsET[interval][0]
                end = self.ROI_ObsET[interval][-1]
                if start <= t <= end:
                    break
                else:
                    continue
        nimages = math.ceil(np.interp(t, self.ROI_ObsET[interval], self.ROI_ObsImg[interval]))
        timeobs = np.interp(t, self.ROI_ObsET[interval], self.ROI_ObsLen[interval])
        # print('t = ', t, '\n obsET = ', self.ROI_ObsET[interval][-1])
        res = np.interp(t, self.ROI_ObsET[interval], self.ROI_ObsRes[interval])
        return nimages, timeobs, res





