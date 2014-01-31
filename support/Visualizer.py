'''
Created on 10.11.2013

@author: Aaron Klein
'''

import matplotlib.pyplot as plt
import math
import numpy as np
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Visualizer():

    def __init__(self, number_of_plots):
        '''
        Constructor
        
        '''
        self._number_of_plots = 2#number_of_plots
        
        fig, self._axarr = plt.subplots(self._number_of_plots, figsize=(20,12))
        fig.subplots_adjust(left=0.04, bottom=0.04, right=0.95, top=0.95,
                    wspace=0.5, hspace=0.5)
        self.fig2, self._axarr2 = plt.subplots(self._number_of_plots, 2, figsize=(20,12))

        #plt.ion()
        #plt.show()
       
    def plot_expected_improvement(self, points, representer_points):
        '''
            points have to be tuples like [(x1,y1),(x2,y2)....]
        '''
        points.sort(key=itemgetter(0))
        a = np.array(points)
        a = a.T
        plt.plot(a[0],a[1],'b')
        func = np.zeros(representer_points.shape[0])
        
        self._axarr[index].plot(representer_points[:,1], func, 'ro')
        plt.draw()
        
    def plot3DSurface(self, model):
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = np.linspace(0, 1, 100)[:, np.newaxis]

        x, y = np.meshgrid(x, y)

        test_inputs = np.zeros((100*100, 2))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                test_inputs[i*100 + j,0] =  x[j][i]
                test_inputs[i*100 + j,1] =  y[j][i]
        (mean, variance) = model.predict(test_inputs, True)

        z = np.zeros((100,100))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                z[j][i] = mean[i*100 + j]
        
        ax = Axes3D(self.fig2)
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='hot')
#         
#         plt.show()
#         #ax.zaxis.set_major_locator(LinearLocator(10))
#         #x.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        self.fig2.colorbar(surf, shrink=0.5, aspect=5)
        plt.draw()
        
    def plot2Dfunction(self, function):
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = np.linspace(0, 1, 100)[:, np.newaxis]

        x, y = np.meshgrid(x, y)

        test_inputs = np.zeros((100*100, 2))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                test_inputs[i*100 + j,0] =  x[j][i]
                test_inputs[i*100 + j,1] =  y[j][i]
        mean = function(test_inputs)

        z = np.zeros((100,100))
        for i in xrange(0, 100):
            for j in xrange(0, 100):
                z[j][i] = mean[i*100 + j]
        
        ax = Axes3D(self.fig2)
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='hot')
#         
#         plt.show()
#         #ax.zaxis.set_major_locator(LinearLocator(10))
#         #x.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        self.fig2.colorbar(surf, shrink=0.5, aspect=5)
        plt.draw()
        plt.show()

    def plot_pmin(self, representer_points, pmin, index):

        heatmap, xedges, yedges = np.histogram2d(representer_points[:,0], representer_points[:,1], bins=10, weights=pmin)

        extent = [0, 1, 0, 1]

        #split the headmaps in two columns
        if(index < 5):
            self._axarr2[index][0].imshow(heatmap, extent=extent)
        else:
            self._axarr2[index-5][1].imshow(heatmap, extent=extent)
        
        plt.show()
        plt.draw()
         
        
    def plot(self, representer_points, X, y, model, index):
        self.plot_projected_gp(X, y, model, index)
        self.plot_representer_points(representer_points, index)
    
    def plot_representer_points(self, representer_points, index):
        '''
            points have to be tuples like [(x1,y1),(x2,y2)....]
        '''
        func = np.zeros(representer_points.shape[0])
        self._axarr[index].plot(representer_points[:,1], func, 'ro')
        plt.draw()


    def plot_projected_gp(self, X, y, model, index):
  
        self._axarr[index].clear()
 
        self._axarr[index].plot(X[:, 1], y, 'g+')
            
        x = np.linspace(0, 1, 100)[:, np.newaxis]

        # all inputs are normalized by spearmint automatically, thus max_size = 1
        test_inputs = np.ones((100, 2))
        for i in range(0, 100):
            test_inputs[i, 1] = x[i]

        (mean, variance) = model.predict(test_inputs, True)

        lower_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            lower_bound[i] = mean[i] - math.sqrt(variance[i])
        
        upper_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            upper_bound[i] = mean[i] + math.sqrt(variance[i])
        

        self._axarr[index].plot(x, mean, 'b')
        self._axarr[index].fill_between(test_inputs[:, 1], upper_bound[:, 0], lower_bound[:, 0], facecolor='red')
        plt.draw()
        
    def plot1Dgp(self, X, y, predict, index=0):
  
        self._axarr[index].clear()
 
        self._axarr[index].plot(X, y, 'g+')
            
        x = np.linspace(0, 1, 100)[:, np.newaxis]

        # all inputs are normalized by spearmint automatically, thus max_size = 1
        test_inputs = np.ones((100, 1))
        for i in range(0, 100):
            test_inputs[i] = x[i]

        (mean, variance) = predict(test_inputs, True)
        lower_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            lower_bound[i] = mean[i] - math.sqrt(variance[i])
        
        upper_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            upper_bound[i] = mean[i] + math.sqrt(variance[i])
        

        self._axarr[index].plot(x, mean, 'b')
        self._axarr[index].fill_between(test_inputs[:,0], upper_bound[:,0], lower_bound[:,0], facecolor='red')
        plt.draw()
        plt.show()