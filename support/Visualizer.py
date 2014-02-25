'''
Created on 10.11.2013

@author: Aaron Klein
'''

import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Visualizer():

    def __init__(self, index):

        self._index_ei = index
        self._performance = index
        self._costs = index
        self._index_entropy = index

    def plot_expected_improvement_one_dim(self, cands, ei_values, best_cands, new_cand):

        fig = plt.figure()
        plt.hold(True)
        plt.plot(cands, ei_values, 'g*')
        plt.plot(best_cands, np.zeros(best_cands.shape[0]), 'b+')
        plt.plot(new_cand, np.zeros(new_cand.shape[0]), 'ro')

        filename = "/home/kleinaa/plots/ei/ei_" + str(self._index_ei) + ".png"

        self._index_ei += 1

        plt.savefig(filename)
        plt.hold(False)

    def plot_expected_improvement_two_dim(self, cands, ei_values, best_cands, new_cand):

        fig = plt.figure()
        plt.hold(True)
        ax = fig.gca(projection='3d')
        ax.plot(cands[:, 0], cands[:, 1], ei_values, 'g*')
        ax.plot(best_cands[:, 0], best_cands[:, 1], np.zeros(best_cands.shape[0]), 'b+')
        ax.plot(new_cand[:, 0], new_cand[:, 1], np.zeros(new_cand.shape[0]), 'ro')

        plt.xlabel('S')
        plt.xlabel('X')

        filename = "/home/kleinaa/plots/ei/ei_" + str(self._index_ei) + ".png"

        self._index_ei += 1

        plt.savefig(filename)
        plt.hold(False)

    def plot_entropy_surface(self, comp, vals, model, cost_model):

        fig = plt.figure()
        plt.hold(True)
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = np.linspace(0, 1, 100)[:, np.newaxis]

        x, y = np.meshgrid(x, y)

        from ..acquisition_functions.entropy_search_big_data import EntropySearchBigData
        entropy_estimator = EntropySearchBigData(comp, vals, model, cost_model)
        entropy = np.zeros([100, 100])
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                entropy[j][i] = entropy_estimator.compute(np.array([x[i][j], y[i][j]]))

        ax = Axes3D(fig)
        ax.plot_surface(x, y, entropy, rstride=1, cstride=1, cmap='hot')

        filename = "/home/kleinaa/plots/entropy/entropy_" + str(self._index_entropy) + ".png"

        self._index_entropy += 1

        plt.savefig(filename)
        plt.hold(False)


    def plot_entropy_one_dim(self, cands, entropy_values):

        fig = plt.figure()
        plt.hold(True)
        plt.plot(cands, entropy_values, 'b.')

        filename = "/home/kleinaa/plots/entropy/entropy_" + str(self._index_entropy) + ".png"

        self._index_entropy += 1

        plt.savefig(filename)
        plt.hold(False)

    def plot_entropy_two_dim(self, cands, entropy_values):

        fig = plt.figure()
        plt.hold(True)
        ax = fig.gca(projection='3d')
        ax.plot(cands[:, 0], cands[:, 1], entropy_values, 'g*')

        filename = "/home/kleinaa/plots/entropy/entropy_" + str(self._index_entropy) + ".png"

        plt.xlabel('S')
        plt.xlabel('X')

        self._index_entropy += 1

        plt.savefig(filename)
        plt.hold(False)

    def plot_gp(self, X, y, model, is_cost):

        fig = plt.figure()
        plt.hold(True)
        plt.plot(X, y, 'g+')
        x = np.linspace(0, 1, 100)[:, np.newaxis]

        test_inputs = np.ones((100, 1))
        for i in range(0, 100):
            test_inputs[i] = x[i]

        (mean, variance) = model.predict(test_inputs, True)
        lower_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            lower_bound[i] = mean[i] - math.sqrt(variance[i])

        upper_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            upper_bound[i] = mean[i] + math.sqrt(variance[i])

        plt.plot(x, mean, 'b')
        plt.fill_between(test_inputs[:, 0], upper_bound[:, 0],
                         lower_bound[:, 0], facecolor='red')
        if is_cost == True:
            filename = "/home/kleinaa/plots/costs/gp_cost_" + str(self._costs) + ".png"
            self._costs += 1
        else:
            filename = "/home/kleinaa/plots/performance/gp_" + str(self._performance) + ".png"
            self._performance += 1

        plt.savefig(filename)
        plt.hold(False)

    def plot_projected_gp(self, X, y, model, is_cost):

        fig = plt.figure()
        plt.hold(True)
        plt.plot(X, y, 'g+')
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        #all inputs are normalized by spearmint automatically, thus max_size =1
        test_inputs = np.ones((100, 2))
        for i in range(0, 100):
            if is_cost == False:
                test_inputs[i, 1] = x[i]
            else:
                test_inputs[i, 0] = x[i]

        (mean, variance) = model.predict(test_inputs, True)
        lower_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            lower_bound[i] = mean[i] - math.sqrt(variance[i])

        upper_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            upper_bound[i] = mean[i] + math.sqrt(variance[i])

        plt.plot(x, mean, 'b')
        if is_cost == False:
            plt.fill_between(test_inputs[:, 1], upper_bound[:, 0],
                         lower_bound[:, 0], facecolor='red')
        else:
            plt.fill_between(test_inputs[:, 0], upper_bound[:, 0],
                         lower_bound[:, 0], facecolor='red')

        if is_cost == True:
            filename = "/home/kleinaa/plots/costs/gp_cost_" + str(self._costs) + ".png"
            self._costs += 1
        else:
            filename = "/home/kleinaa/plots/performance/gp_" + str(self._performance) + ".png"
            self._performance += 1

        plt.savefig(filename)
        plt.hold(False)

    def plot_two_dim_gp(self, X, y, model, is_cost):

        fig = plt.figure()
        plt.hold(True)

        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = np.linspace(0, 1, 100)[:, np.newaxis]
 
        x, y = np.meshgrid(x, y)
 
        test_inputs = np.zeros((100 * 100, 2))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                test_inputs[i * 100 + j, 0] =  x[j][i]
                test_inputs[i * 100 + j, 1] =  y[j][i]
        (mean, variance) = model.predict(test_inputs, True)
 
        z = np.zeros((100, 100))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                z[j][i] = mean[i * 100 + j]
 
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='hot')

        if is_cost == True:
            filename = "/home/kleinaa/plots/costs/gp_cost_" + str(self._costs) + ".png"
            self._costs += 1
        else:
            filename = "/home/kleinaa/plots/performance/gp_" + str(self._performance) + ".png"
            self._performance += 1

        plt.savefig(filename)
        plt.hold(False)

    def plot_representer_points_one_dim(self, representer_points):

        fig = plt.figure()
        plt.hold(True)
        plt.plot(representer_points, np.zeros(representer_points.shape[0]), 'ro')

        filename = "/home/kleinaa/plots/entropy/entropy_" + str(self._index_entropy) + ".png"
        self._index_entropy += 1

        plt.savefig(filename)
        plt.hold(False)

#     def plot3DSurface(self, model):
#         x = np.linspace(0, 1, 100)[:, np.newaxis]
#         y = np.linspace(0, 1, 100)[:, np.newaxis]
# 
#         x, y = np.meshgrid(x, y)
# 
#         test_inputs = np.zeros((100*100, 2))
#         for i in xrange(0, 100):
#             for j in xrange(0, 100):
#                 test_inputs[i*100 + j,0] =  x[j][i]
#                 test_inputs[i*100 + j,1] =  y[j][i]
#         (mean, variance) = model.predict(test_inputs, True)
# 
#         z = np.zeros((100,100))
#         for i in xrange(0, 100):
#             for j in xrange(0, 100):
#                 z[j][i] = mean[i*100 + j]
# 
#         ax = Axes3D(self.fig2)
#         surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='hot')
#         self.fig2.colorbar(surf, shrink=0.5, aspect=5)
#         plt.draw()
#         
#     def plot2Dfunction(self, function):
#         x = np.linspace(0, 1, 100)[:, np.newaxis]
#         y = np.linspace(0, 1, 100)[:, np.newaxis]
# 
#         x, y = np.meshgrid(x, y)
# 
#         test_inputs = np.zeros((100*100, 2))
#         for i in xrange(0, 100):
#             for j in xrange(0, 100):
#                 test_inputs[i*100 + j,0] =  x[j][i]
#                 test_inputs[i*100 + j,1] =  y[j][i]
#         mean = function(test_inputs)
# 
#         z = np.zeros((100,100))
#         for i in xrange(0, 100):
#             for j in xrange(0, 100):
#                 z[j][i] = mean[i*100 + j]
#         
#         fig = plt.figure()
#         ax = Axes3D(self.fig2)
#         surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='hot')
# #         
# #         plt.show()
# #         #ax.zaxis.set_major_locator(LinearLocator(10))
# #         #x.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#         
#         self.fig2.colorbar(surf, shrink=0.5, aspect=5)
#         plt.draw()
#         plt.show()
# 
#     def plot_pmin(self, representer_points, pmin, index):
# 
#         heatmap, xedges, yedges = np.histogram2d(representer_points[:,0], representer_points[:,1], bins=10, weights=pmin)
# 
#         extent = [0, 1, 0, 1]
# 
#         #split the headmaps in two columns
#         if(index < 5):
#             self._axarr2[index][0].imshow(heatmap, extent=extent)
#         else:
#             self._axarr2[index-5][1].imshow(heatmap, extent=extent)
#         
#         plt.show()
#         plt.draw()
#          
#         
#     def plot(self, representer_points, X, y, model, index):
#         self.plot_projected_gp(X, y, model, index)
#         self.plot_representer_points(representer_points, index)
#     
# 
# 
#     def plot1Dgp(self, X, y, predict, index=0):
#   
#         self._axarr[index].clear()
#  
#         self._axarr[index].plot(X, y, 'g+')
#             
#         x = np.linspace(0, 1, 100)[:, np.newaxis]
# 
#         # all inputs are normalized by spearmint automatically, thus max_size = 1
#         test_inputs = np.ones((100, 1))
#         for i in range(0, 100):
#             test_inputs[i] = x[i]
# 
#         (mean, variance) = predict(test_inputs, True)
#         lower_bound = np.zeros((100, 1))
#         for i in range(0, mean.shape[0]):
#             lower_bound[i] = mean[i] - math.sqrt(variance[i])
#         
#         upper_bound = np.zeros((100, 1))
#         for i in range(0, mean.shape[0]):
#             upper_bound[i] = mean[i] + math.sqrt(variance[i])
#         
# 
#         self._axarr[index].plot(x, mean, 'b')
#         self._axarr[index].fill_between(test_inputs[:,0], upper_bound[:,0], lower_bound[:,0], facecolor='red')
#         plt.draw()
#         plt.show()