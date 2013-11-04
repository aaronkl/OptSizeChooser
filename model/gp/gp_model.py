'''
Created on Oct 28, 2013

@author: Simon Bartels
'''

import gp
import numpy as np
import scipy.linalg as spla
from model.model import Model


def _get_cholesky(cov, X, noise, ls):
    # Compute the required Cholesky.
    K = cov(ls, X) + noise * np.eye(X.shape[0])
    L = spla.cholesky(K, lower=True)
    return L

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
        
        self._X = X
        self._y = y
        self._covar = getattr(gp, covarname)
        self._covar_derivative = getattr(gp, "grad_" + covarname)
        self._ls = np.ones(X.shape[1])
        self._amp2 = 1.0
        self._noise = 1e-3
        self._mean = self.mean = np.mean(y)
        #the Cholesky of the correlation matrix
        self._L = _get_cholesky(self._covar, self._X, self._noise, self._ls)
        self._alpha = spla.cho_solve((self._L, True), self._y - self._mean)
        
    def predict(self, Xstar, variance=False):
        kXstar = self._covar(self._ls, self._X, Xstar)
        func_m = np.dot(kXstar.T, self._alpha) + self.mean
        if not variance:
            return func_m
        
        beta = spla.solve_triangular(self._L, kXstar, lower=True)
        func_v = self._amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        return (func_m, func_v)
        
    def getGradients(self, xstar):
        #TODO: Might be that sign of mean gradient is wrong!
        xstar = np.array([xstar])
        # gradient of mean
        # dk(X,x*) / dx
        dk = self._amp2 * self._covar_derivative(self._ls, xstar, self._X)[0]
        grad_m = np.dot(dk.T, self._alpha)
        
        # gradient of variance
        k = self._amp2 * self._covar(self._ls, xstar, self._X)
        #kK = k^T * K^-1
        kK = spla.cho_solve((self._L, True), k.T)
        (_, v) = self.predictVariance(xstar)
        grad_v = -np.dot(kK.T, dk) / np.sqrt(v[0])
        #As in spear mint grad_v is of the form [[v1, v2, ...]]
        #TODO: Check if this is really necessary. Seems dirty.
        return (grad_m, grad_v)
    
    def optimize(self):
        #TODO: implement
        raise NotImplementedError("Not implemented yet!")
        