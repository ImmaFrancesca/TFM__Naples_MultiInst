import sys

from DataManager import DataManager
from oROI import ROI
from osample import sample
import matplotlib.pyplot as plt
from PMOT.ooaga import aga
import spiceypy as spice
import spiceypy.utils.support_types as stypes
from ooamaga import amaga

twList = []
# ROIs tw instrument 1:
ROI1_i1tw = [[1., 3.], [5., 8.], [13., 18.]]
ROI2_i1tw = [[4., 9.], [15., 20.]]
ROI3_i1tw = [[0.5, 7], [8., 14],[19,20]]
ROI4_i1tw = [[1.,5.],[6.,8.],[15.,18.]]
ROI5_i1tw = [[3.4,6],[8.,9.5]]
twList.append([ROI1_i1tw, ROI2_i1tw, ROI3_i1tw, ROI4_i1tw, ROI5_i1tw])

# ROIs tw instrument 2:
ROI1_i2tw = [[0., 3.], [7., 11.], [14., 19.]]
ROI2_i2tw = [[6.,9.],[10.,12.],[15.,18.]]
ROI3_i2tw = [[0.5,3.]]
ROI4_i2tw = [[1.,5.],[8.,11.],[15.,18.]]
twList.append([ROI1_i2tw, ROI2_i2tw, ROI3_i2tw, ROI4_i2tw])

# ROIs tw instrument 3:
ROI1_i3tw = [[2., 13.]]
ROI2_i3tw = [[1., 6.], [14., 19.]]
ROI3_i3tw = [[5., 8.], [17., 20.]]
ROI4_i3tw = [[1.,4.],[8.,9.]]
ROI5_i3tw = [[7.,13.],[18.,20.]]
twList.append([ROI1_i3tw, ROI2_i3tw, ROI3_i3tw, ROI4_i3tw, ROI5_i3tw])

# ROIs tw instrument 4:
ROI1_i4tw = [[0.5,2.],[7.,8.9],[10.,12.5]]
ROI2_i4tw = [[4.,6.],[7.5,9.],[10.,11.9]]
ROI3_i4tw = [[0.,2.],[2.5,6.],[8.,10.77],[13.,15.]]
ROI4_i4tw = [[4.5,8.],[8.5,10.],[17.5,19.5]]
ROI5_i4tw = [[0.5,3.7],[8.,12.],[15.,19.]]
twList.append([ROI1_i4tw, ROI2_i4tw, ROI3_i4tw, ROI4_i4tw, ROI5_i4tw])

ROIlist = []
for instrument in range(len(twList)):
    instList = []
    instrumentROIstw = twList[instrument]
    for i, roitw in enumerate(instrumentROIstw):
        roiName = 'ROI' + str(i +1)
        twroi = stypes.SPICEDOUBLE_CELL(2000)
        for interval in roitw:
            spice.wninsd(interval[0], interval[1], twroi)
        instList.append(ROI(roiName, twroi))
    ROIlist.append(instList)

DataManager(ROIlist)
plan1 = sample(len(ROIlist))
feasibility = plan1.checkFeasibility()

if not feasibility:
    sys.exit()
plan1.ranFun()
plan1.plotGantt()
print('initial fitness = ', plan1.fitFun())
plan1.mutFun()
plan1.fitFun(True)
plan1.plotGantt()

plan2 = sample(len(ROIlist))
plan2.ranFun()
plan2.plotGantt()

plan1.repFun(plan2, 0, 0)
plan1.plotGantt()

"""
myaga = aga(plan1, 100)
bestInd, bestFit, bestType, glast = myaga.run(500)
bestInd.plotGantt()
bestInd.fitFun(True)
print('bestFit = ', bestFit)
print('bestType = ', bestType)
print('generation = ', glast)
"""

myaga = amaga(plan1,50)
myaga.run(2)
#myaga.plotPopulation2d()
#plt.show()
#myaga.plotSatus2d()
#plt.show()
myaga.printStatus()
#myaga.pop[0].plotGantt()


