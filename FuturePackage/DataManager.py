import copy
import numpy as np
from sys import getsizeof
import tracemalloc
from webbrowser import Error


# PONER COMENTARIOS
class DataManager:
    __instance = None
    __lock = False

    @staticmethod
    def getInstance():
        if DataManager.__instance is None:
            DataManager()
        return DataManager.__instance

    def __init__(self, ROIlist, instrumentsList, observer):
        if ROIlist is not None:
            if DataManager.__lock:
                raise Exception("Already initialized")
            self.ROIlist = ROIlist
            self.instrumentsList = instrumentsList
            self.observer = observer
            self.maxRes = [None] * len(self.instrumentsList)
            self.minRes = [None] * len(self.instrumentsList)
            self.getMaxMinRes()
            DataManager.__lock = True
        DataManager.__instance = self

    def getROIList(self,s = None, e = None):
        if s is None:
            return self.ROIlist
        else:
            return [inst[s[i]:e[i]+1] for i,inst in enumerate(self.ROIlist)]

    def getMaxMinRes(self):
        if self.maxRes == [None] * len(self.instrumentsList) or self.minRes == [None] * len(self.instrumentsList):
            for i, instrument in enumerate(self.instrumentsList):
                if instrument.type == 'CAMERA':
                    min_res = np.inf
                    max_res = -  np.inf
                    for roi in self.ROIlist[i]:
                        min_res_roi = min(np.concatenate(roi.ROI_ObsRes))
                        max_res_roi = max(np.concatenate(roi.ROI_ObsRes))
                        if min_res_roi < min_res: min_res = copy.deepcopy(min_res_roi)
                        if max_res_roi > max_res: max_res = copy.deepcopy(max_res_roi)
                    self.maxRes[i] = copy.deepcopy(max_res)
                    self.minRes[i] = copy.deepcopy(min_res)
        else:
            return self.maxRes, self.minRes

    def getSingleROI(self, i, instrument_index = None):

        if instrument_index == None:
            single_ROIs = []
            for inst in self.ROIlist:
                try:
                    single_ROIs.append(inst[i])
                except:
                    raise Exception('The instrument at index',self.ROIlist.index(inst),'does not have a ROI n.',i)
            return [inst[i] for inst in self.ROIlist]
        else:
            return self.roiList[instrument_index][i]

    def getInstrumentData(self):
        return self.instrumentsList

    def getObserver(self):
        return self.observer

    #def initObservationDataBase(self, twL, i = None):
    #    if i is None: #i is the roi number, if None it is assumed that a sorted list containing the constrained TW for each roi is being passed. It must have the same order as the roiList
    #        for i, tw in enumerate(twL):
    #            self.roiList1[i].initializeObservationDataBase(tw, self.instrument, self.observer)
    #    else:
    #        self.roiList1[i].initializeObservationDataBase(twL, self.instrument, self.observer)