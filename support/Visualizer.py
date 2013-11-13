'''
Created on 10.11.2013

@author: Aaron Klein
'''

import matplotlib.pyplot as plt
import math
import numpy as np
from model.gp.gp_model_factory import GPModelFactory


class Visualizer():
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    def plot_acquisition_function(self, input_values, func_values):
        plt.figure(1)

        plt.plot(input_values,func_values,'b+')
        plt.show()
        
    '''
        Plots the project Gaussian Process
    '''
    def plot_projected_gp(self, X, y, model):
        
        plt.figure(2)
        plt.plot(X[:, 1], y, 'g+')

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
        
        plt.plot(x, mean, 'b')
        plt.fill_between(test_inputs[:, 1], upper_bound[:, 0], lower_bound[:, 0], facecolor='red')
        plt.grid(True)

        plt.show()