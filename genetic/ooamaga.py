import os

import matplotlib
import numpy as npy
import copy
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from PMOT.ooaga import aga

class amaga(aga):

    def __init__(self, sample, pop, options=None):
        super(amaga, self).__init__(sample, pop, options)
        self.front = None
        self.sorted = False
        self.fit=npy.zeros([self.getPopulationSize(),self.sample.getNgoals()]) # fit[individual,goal]

    def evalFitness(self):
        self.fit=npy.zeros([self.getPopulationSize(),self.sample.getNgoals()])
        for i in range(self.getPopulationSize()):
            self.fit[i,:]=self.pop[i].fitFun()

    def euclideanDistance(self, a, b):
        #print(npy.linalg.norm(a - b))
        return npy.linalg.norm(a - b)
    
    def buildFronts(self):
        NFRONT = 10
        self.front=[None]*self.getPopulationSize()

        for f in range(NFRONT):
            for i in range(self.getPopulationSize()): # shall we assign individual i to front f ?
                if self.front[i] is not None: # skip  individuals already assigned to a front
                    continue

                if (self.fit[i][0] > 1000) and (self.fit[i][1] > 1000): # PROBLEM_CHANGE : if we have different instruments, this must be changed
                    continue

                dominant=True

                for ii in range(self.getPopulationSize()): # for all other individuals ...
                    if i == ii:
                        continue

                    if (self.front[ii] is not None) and (self.front[ii]<f): # except those assigned to better fronts
                        continue

                    better_in_all = all(self.fit[ii, k] <= self.fit[i, k] for k in range(self.sample.getNgoals())) # best fit in all objectives
                    
                    better_in_at_least_one = False # any(self.fit[ii, k] < self.fit[i, k] for k in range(self.sample.getNgoals()))
                    n_equal = 0
                    n_better = 0
                    for k in range(self.sample.getNgoals()):
                        if self.fit[ii, k] ==  self.fit[i, k]: # counts how many times they have the same fitness
                            n_equal += 1
                        elif self.fit[ii, k] <  self.fit[i, k]: # counts how many times is better
                            n_better += 1
                        
                    if (n_equal + n_better) == self.sample.getNgoals() and n_better > 0: # if is equal in all of the obectives and better in AT LEAST one
                        better_in_at_least_one = True
                    else: # better in one and worse in other? non dominant, could be same pareto or not
                        continue
                        
                    if better_in_all or better_in_at_least_one: #si può eliminare, perchè la parte di codice sotto è eseguita solo se sono vere le due condizioni (in particolare la seconda, che implica la prima)
                        dominant = False
                        break

                if dominant:
                    self.front[i]=f
                    #print('i=',i,'assigned to front f=',f)

        for i in range(self.getPopulationSize()):
            if self.front[i] is None: self.front[i]=NFRONT+1

        self.sorted=False

    def buildCrowds(self):
        NFRONT=10
        self.crowd=[0]*self.getPopulationSize()
        for f in range(NFRONT):
            front_index = [i for i in range(self.getPopulationSize()) if self.front[i] == f] # Gets individuals from pareto f

            if len(front_index) == 0:
                continue

            for m in range(self.sample.getNgoals()): # For each objective
                aux = 0
                for i, index1 in enumerate(front_index):
                    for ii, index2 in enumerate(front_index): # Order the indexes depending on the fitness
                        if self.fit[index2][m] < self.fit[index1][m]:
                            aux = front_index[ii]
                            front_index[ii] = front_index[i]
                            front_index[i] = aux

                #front_index.sort(key=lambda i: self.fit[i][m])

                self.crowd[front_index[0]] = float('inf') # Give infinite to first and last individuals ordered by their fitness (greatest diversity)
                self.crowd[front_index[-1]] = float('inf')

                norm_range = self.fit[front_index[-1]][m] - self.fit[front_index[0]][m] # For each objective it will be different
                if norm_range == 0:
                    norm_range = 1

                for j in range(1, len(front_index) - 1):
                    next_fit = self.fit[front_index[j+1]][m]
                    prev_fit = self.fit[front_index[j-1]][m]
                    self.crowd[front_index[j]] += ((next_fit - prev_fit) / norm_range) ** 2
        
        self.crowd = npy.sqrt(self.crowd)

    def buildCrowdsEuclidean(self):
        NFRONT=10
        self.crowd=[0] * self.getPopulationSize()
        self.fitness = [0] * self.getPopulationSize()
        for f in range(NFRONT):
            front_index = [i for i in range(self.getPopulationSize()) if self.front[i] == f] # Gets individuals from pareto f

            if len(front_index) == 0:
                continue

            for m in range(self.sample.getNgoals()): # For each objective
                for j in range(len(front_index)):
                    self.fitness[front_index[j]] += self.fit[front_index[j]][m] ** 2

            self.fitness = npy.sqrt(self.fitness)

            aux = 0
            for i, index1 in enumerate(front_index):
                for ii, index2 in enumerate(front_index): # Order the indexes depending on the fitness
                    if self.fitness[index2] < self.fitness[index1]:
                        aux = front_index[ii]
                        front_index[ii] = front_index[i]
                        front_index[i] = aux

            #front_index.sort(key=lambda i: self.fit[i][m])

            self.crowd[front_index[0]] = float('inf') # Give infinite to first and last individuals ordered by their fitness (greatest diversity)
            self.crowd[front_index[-1]] = float('inf')

            for j in range(1, len(front_index) - 1):
                next_fit = self.fitness[front_index[j+1]]
                prev_fit = self.fitness[front_index[j-1]]
                self.crowd[front_index[j]] += npy.abs(next_fit - prev_fit)

    def getFrontSize(self,f):
        return sum(self.front[i]==f for i in range(self.getPopulationSize()))
    
    def sortByFronts(self):
        sorted_indices=sorted(range(len(self.pop)), key=lambda k: self.front[k])
        self.pop= [self.pop[i] for i in sorted_indices]
        self.front = [self.front[i] for i in sorted_indices]
        q=npy.zeros([self.getPopulationSize(),self.sample.getNgoals()]) # fit[individual,goal]
        for i in range(len(self.pop)): # for i in sorted_indices:
            q[i,:]=self.fit[sorted_indices[i],:]
        self.fit=q
        self.sorted=True
        self.setOption('sortBy','Pareto Front')

    def sortByCrowds(self):
        sorted_indices=sorted(range(len(self.pop)), key=lambda k: self.front[k])
        start = 0
        while start < len(self.pop): # Sorts pareto fronts population depending on the crowding distance
            end = start
            while end < len(self.pop) and self.front[sorted_indices[start]] == self.front[sorted_indices[end]]:
                end += 1
            sorted_indices[start:end] = sorted(sorted_indices[start:end], key=lambda k: self.crowd[k], reverse=True) # From start to end of each front
            start = end
        self.pop= [self.pop[i] for i in sorted_indices]
        self.front = [self.front[i] for i in sorted_indices]
        self.crowd = [self.crowd[i] for i in sorted_indices]
        q=npy.zeros([self.getPopulationSize(),self.sample.getNgoals()]) # fit[individual,goal]
        for i in range(len(self.pop)): # for i in sorted_indices:
            q[i,:]=self.fit[sorted_indices[i],:]
        self.fit=q
        self.sorted=True
        self.setOption('sortBy','Crowding Distance')

    def evalFitnessAndSort(self, sortByCrowding = True): # eval fitness and build Pareto fronts
        self.evalFitness()
        self.buildFronts()
        if sortByCrowding:
            self.buildCrowds()
            self.sortByCrowds()
        else:
            self.sortByFronts()

    def mutateDegenerates(self,gen):
        if not self.sorted:
            raise Exception('sort to detect degenerates')
        forcedMutants=[]
        for i in range(1,len(self.pop)): # distance between consecutive individuals
            dist=self.pop[i].distance(self.pop[i-1])
            if dist<= self.options['degenerateDistance']: # if too close, force to mutate
                if self.options['info']>1: print('Distance = ',dist,'Individual ',i,'=',self.pop[i],' forced to mutate and reevaluated')
                forcedMutants.append(i)

        if self.options['info']>1:
            print('forced mutants:',forcedMutants)

        for i in forcedMutants:
                q= copy.deepcopy(self.pop[i])
                q.mutFun(self.fit[i,:],gen)
                self.pop[i]=q
                self.fit[i,:]=q.fitFun()
                self.popType[i]='FM' # forced mutation
        self.buildFronts()
        if self.options['sortBy'] == 'Crowding Distance':
            self.buildCrowds()
            self.sortByCrowds()
        elif self.options['sortBy'] == 'Pareto Front':
            self.sortByFronts()

 #   def repopulate(self,gen):
 #       if not self.sorted: raise Exception('Build fronts and sort first')
 #       pop2 = self.pop
 #       for i in range(len(self.pop)):
 #           if self.front[i] <=1: # unchanged elite
 #               pop2[i]=copy.deepcopy(self.pop[i])

    def metricM2(self, sigma = 1.0):
        count = 0
        pareto_front = [self.fit[i, :] for i in range(self.getFrontSize(0))]
        #print(pareto_front)
        for ind1 in range(len(pareto_front)):
            for ind2 in range(ind1 + 1, len(pareto_front)):
                if self.euclideanDistance(pareto_front[ind1], pareto_front[ind2]) > sigma:
                    count += 1
        return count / (len(pareto_front)-1)
    
    def frontSpread(self):
        pareto_front = [self.fit[i, :] for i in range(self.getFrontSize(0))]
        max_distance = [0] * len(pareto_front[0])
        for n in range(len(pareto_front[0])):
            max_distance[n] = npy.max([npy.abs(ind1[n] - ind2[n]) for ind1 in pareto_front for ind2 in pareto_front])
        return npy.sqrt(npy.sum(max_distance))

    def convergenceMetric(self, pareto_optimal): # Better as much lower as possible
        pareto_front = [self.fit[i, :] for i in range(self.getFrontSize(0))]
        total = 0
        for p in pareto_front:
            min_distance = npy.min([npy.linalg.norm(p - xp) for xp in pareto_optimal])
            total += min_distance
        gamma = total / self.getFrontSize(0)
        return gamma

    def printStatus(self):
        print('sorted =',self.sorted)
        for i in range(self.getPopulationSize()):
            print('i =', f'{i:3d}', ':', self.pop[i],' front =',self.front[i],' fit =',end='')
            for g in range(self.sample.getNgoals()):
                print(f'{self.fit[i,g]}',' ',end='')
            if i>0:
                print(' distToPrevious:',self.pop[i].distance(self.pop[i-1]))
            else:
                print()

    def plotSatus2d(self):
        # col = {0: 'r', 1: 'y', 2: 'b',3:'m',4:'c',5:'g',6:'k',7:'k',8:'k',9:'k'}
        # mar = {0: 'o', 1: 'o', 2: 'o', 3:'o', 4:'o', 5:'o', 6:'o', 7:'o', 8:'o', 9:'o'}
        matplotlib.use('Agg')
        cmap = plt.cm.coolwarm
        unique_fronts = sorted(set(self.front))  # Get sorted unique fronts
        norm = Normalize(vmin=min(unique_fronts), vmax=max(unique_fronts))
        # front_colors = {front: cmap(i / len(unique_fronts)) for i, front in enumerate(unique_fronts)}
        fig, ax = plt.subplots()
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax = ax)
        cbar.set_label("# Pareto Front")  # Etichetta barra colore

        front_legend_map = []

        for i in range(self.getPopulationSize()):
            if self.front[i] > 10:
                continue
            # if self.fit[i][0] > 10:
            #   continue
            else:
                # c = front_colors.get(self.front[i], 'k')
                c = cmap(norm(self.front[i]))
            plt.plot(self.pop[i].fitFun()[0], self.pop[i].fitFun()[1], 'o', color=c, markersize=5)
            # print(f'self.fit: {self.fit[i]}')

        # plt.plot([], [], 'o', color = c, label='Pareto Front', markersize=5.0)
        plt.title('Objective Space', fontweight='bold')
        # plt.legend()
        # plt.legend(handles=front_legend_map, loc='center left', bbox_to_anchor=(1.0, 0.5))
        # plt.xticks(npy.arange(0, 11, 1))
        # plt.yticks(npy.arange(0, 11, 1))
        # plt.xlim([0.4, 1])
        # plt.ylim([1e+7, 5e+8])


    def plotPopulation2d(self):
        # Use a colormap for gradient colors
        cmap = plt.cm.coolwarm
        unique_fronts = sorted(set(self.front))  # Get sorted unique fronts
        front_colors = {front: cmap(i / len(unique_fronts)) for i, front in enumerate(unique_fronts)}

        front_legend_map = []

        for i in range(self.getPopulationSize()):
            if self.front[i] > 10:
                continue
            c = front_colors.get(self.front[i], 'k')  # Get color for current front
        
            if self.front[i] not in [patch.get_label().split()[-1] for patch in front_legend_map]:
                front_legend_map.append(Patch(facecolor=c, label=f'P{i} Front {self.front[i]}'))
                plt.plot(self.pop[i].getVector()[0], self.pop[i].getVector()[1], 'o', color=c)
            else:
                plt.plot(self.pop[i].getVector()[0], self.pop[i].getVector()[1], 'o', color=c)


            #fitness_text = f'{self.fit[i]}'
            #if i < 10:
            #    plt.text(self.pop[i].getVector()[0] + 0.2, self.pop[i].getVector()[1] + 0.5, fitness_text, color='black', fontsize=8)
            #else:
            #    if self.front[i] == 0:
            #        plt.text(self.pop[i].getVector()[0] - 1.75, self.pop[i].getVector()[1] + 0.5, fitness_text, color='black', fontsize=8)
            #    else:
            #        plt.text(self.pop[i].getVector()[0] - 5, self.pop[i].getVector()[1] + 0.5, fitness_text, color='black', fontsize=8)

            
            #if i > 0:
            #    arrow = patches.FancyArrowPatch(
            #    (self.pop[i-1].getVector()[0], self.pop[i-1].getVector()[1]),
            #    (self.pop[i].getVector()[0], self.pop[i].getVector()[1] - 0.1),
            #    connectionstyle="arc3,rad=-0.6",  # This creates a curved arrow
            #    color=c,
            #    arrowstyle='-|>',
            #    mutation_scale=10,
            #    lw=1
            #    )
            #    plt.gca().add_patch(arrow)
        # Sort legend patches by front
        front_legend_map = sorted(front_legend_map, key=lambda patch: int(patch.get_label().split()[-1]))
        plt.plot([], [], 'o', color= c, label='Pareto Front', markersize=5.0)

        plt.axis('equal')
        plt.title('Population Space', fontweight='bold')
        #plt.legend(handles=front_legend_map, loc='center left', bbox_to_anchor=(1.0, 0.5))
        #plt.xticks(npy.arange(-8, 9, 1))
        #plt.yticks(npy.arange(-1, 12, 1))
        #plt.ylim(-1, 11)
        plt.show()
    def run(self, ng=100, sortByCrowding = True):
        print('AGA vanilla run') if self.options['info'] > 0 else None
        g_last = 0
        for g in range(ng):
            print('generation n.',g)
            self.evalFitnessAndSort(sortByCrowding)
            self.plotSatus2d()
            plt.title(f'Multi-Instrument Schedule Optimization. Generation {g}', fontweight='bold', fontsize=14, y=1.05)
            plt.xlabel('CAMERA 1 Fitness', fontweight='bold', fontsize=12)
            plt.ylabel('CAMERA 2 Fitness', fontweight='bold', fontsize=12)
            plt.xticks(fontsize=12, rotation=45)
            plt.yticks(fontsize=12)
            plt.xlim([0.12, 0.26])
            plt.ylim([0.12, 0.26])
            plt.grid(True, 'major')
            plt.tight_layout()
            plt.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.20)
            # plt.plot([], [], 'o', color= 'b', label='Pareto Front', markersize=5.0)
            # plt.legend()
            homeFolder = os.path.expanduser('~')
            dataFolder = os.path.join(homeFolder, 'ParetoMultiIns')
            if not os.path.isdir(dataFolder):
                os.makedirs(dataFolder, exist_ok=True)
            plt.savefig(f'{dataFolder}/Fronts_{g}')
            plt.close()
            self.mutateDegenerates(g)
            self.repopulate(g)
            g_last = g + 1
        self.evalFitnessAndSort(sortByCrowding)
        self.plotSatus2d()
        plt.title(f'Multi-Instrument Schedule Optimization. Generation {g}', fontweight='bold', fontsize=14, y=1.05)
        plt.xlabel('CAMERA 1 Fitness', fontweight='bold', fontsize=12)
        plt.ylabel('CAMERA 2 Fitness', fontweight='bold', fontsize=12)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.xlim([0.12, 0.26])
        plt.ylim([0.12, 0.26])
        plt.grid(True, 'major')
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.20)
        # plt.plot([], [], 'o', color= 'b', label='Pareto Front', markersize=5.0)
        # plt.legend()
        homeFolder = os.path.expanduser('~')
        dataFolder = os.path.join(homeFolder, 'ParetoMultiIns')
        if not os.path.isdir(dataFolder):
            os.makedirs(dataFolder, exist_ok = True)
        plt.savefig(f'{dataFolder}/Fronts_{g}')
        plt.close()

        self.plotSatus2d()
        plt.title(f'Multi-Instrument Schedule Optimization. Generation {g}', fontweight='bold', fontsize=14, y=1.05)
        plt.xlabel('CAMERA 1 Fitness', fontweight='bold', fontsize=12)
        plt.ylabel('CAMERA 2 Fitness', fontweight='bold', fontsize=12)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.grid(True, 'major')
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, right=0.99, top=0.9, bottom=0.20)
        # plt.plot([], [], 'o', color= 'b', label='Pareto Front', markersize=5.0)
        # plt.legend()
        homeFolder = os.path.expanduser('~')
        dataFolder = os.path.join(homeFolder, 'ParetoMultiIns')
        if not os.path.isdir(dataFolder):
            os.makedirs(dataFolder, exist_ok=True)
        plt.savefig(f'{dataFolder}/Fronts_{g}_zoom')
        plt.close()







# Test example
#if __name__ == "__main__":
#
#    npy.random.seed(12345)
#
#    q=mgbase(npy.array([0.,0.]))
#    q.setVector(npy.array([0,1]))
#
#    print(q.fitFun())
#
#    mymaga=amaga(q,400)
#    mymaga.setOption('nd',0)
#
#    mymaga.setOption('ne', 100)
#    mymaga.setOption('nm', 40)
#
#    for q in range(200):
#        print('q=',q)
#        mymaga.evalFitness()
#        mymaga.buildFronts()
#        mymaga.sortByFronts()
#        #mymaga.printStatus()
#        #mymaga.plotPopulation2d()
#
#        mymaga.repopulate(q)
#
#    mymaga.plotPopulation2d()
#    plt.show()
#    mymaga.plotSatus2d()
#    plt.show()
#
#    exit(0)
#    print('front0 size= ',mymaga.getFrontSize(0))
#    mymaga.setOption('ne',mymaga.getFrontSize(0))
#
#    mymaga.repopulate(0)
#    mymaga.evalFitness()
#    mymaga.buildFronts()
#    mymaga.sortByFronts()
#    mymaga.printStatus()
#    #mymaga.plotSatus2d()
#    mymaga.plotPopulation2d()
#    plt.show()
