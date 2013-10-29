'''
Created on Oct 28, 2013

@author: Simon Bartels
'''

import gp
import numpy as np
from model.model import Model

class GPModel(Model):
    '''
    classdocs
    '''


    def __init__(self, X, y, covarname):
        #TODO: just a stub
        '''
        Constructor
        Args:
            X: The observed inputs.
            y: The corresponding observed values.
        '''
        
        self.X = X
        self.y = y
        self.covar = getattr(gp, covarname)
        self.covar_derivative = getattr(gp, "grad_" + covarname)
        self.ls = np.ones(X.shape[1])
        self.amp2 = 1.0
        self.Kinv #inverse correlation matrix
        
    def getGradients(self, xstar):
        #TODO: check if this works - xstar is a vector and X a matrix!
        k = self.amp2 * self.covar_derivative(self.ls, xstar, self.X)
        #kK = k^T * K^-1
        kK = np.dot(k.T, self.Kinv)
        (_, v) = self.predictVariance(np.array([xstar]))
        grad_m = np.dot(kK, self.y)
        grad_v = np.dot(kK, k) / v[0]
        return (grad_m, grad_v)