'''
Created on Oct 28, 2013

@author: Simon Bartels
'''

import gp
import numpy as np
import scipy.linalg as spla
from model.model import Model


def _get_covariance_matrix(cov, X, noise, ls):
    # Compute the required Cholesky.
    K = cov(ls, X) + noise * np.eye(X.shape[0])
    Kinv = spla.cholesky(K, lower=True)
    return Kinv

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
        #inverse correlation matrix
        self._Kinv = _get_covariance_matrix(self._covar, self._X, self._noise, self._ls)
        
    def predict(self, Xstar, variance=False):
        kXstar = self._covar(self._ls, self._X, Xstar)
        # Predictive things.
        # Solve the linear systems.
        alpha = spla.cho_solve((self._Kinv, True), self._y - self._mean)
        # Predict the marginal means
        func_m = np.dot(kXstar.T, alpha) + self.mean
        if not variance:
            return func_m
        
        beta = spla.solve_triangular(self._Kinv, kXstar, lower=True)
        func_v = self._amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        return (func_m, func_v)
        
    def getGradients(self, xstar):
        #TODO: check if this works - xstar is a vector and X a matrix!
        k = self._amp2 * self._covar_derivative(self._ls, xstar, self._X)[0]
        #kK = k^T * K^-1
        kK = np.dot(k.T, self._Kinv)
        (_, v) = self.predictVariance(xstar)
        grad_m = np.dot(kK, self._y)
        grad_v = np.dot(kK, k) / v[0]
        return (grad_m, grad_v)
    
    def optimize(self):
        #TODO: implement
        raise NotImplementedError("Not implemented yet!")
        