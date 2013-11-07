'''
Created on Oct 28, 2013

@author: Simon Bartels
'''

import gp
import numpy as np
import scipy.linalg as spla
from model.model import Model



class GPModel(Model):
    '''
classdocs
'''

    def __init__(self, X, y, mean, noise, amp2, ls, covarname="Matern52"):
        #TODO: just a stub
        '''
Constructor
Args:
X: The observed inputs.
y: The corresponding observed values.
'''
        
        self._X = X
        self._y = y
        self._cov_func = getattr(gp, covarname)
        self._covar_derivative = getattr(gp, "grad_" + covarname)
        self._ls = ls
        self._amp2 = amp2
        self._mean = mean
        self._noise = noise
        #the Cholesky of the correlation matrix
        self._K = self._compute_covariance(X) + self._noise * np.eye(self._X.shape[0])
        self._L = spla.cholesky(self._K, lower=True)
        self._alpha = spla.cho_solve((self._L, True), self._y - self._mean)
        
    def predict(self, Xstar, variance=False):
        kXstar = self._compute_covariance(self._X, Xstar)
        func_m = np.dot(kXstar.T, self._alpha) + self._mean
        if not variance:
            return func_m
        
        beta = spla.solve_triangular(self._L, kXstar, lower=True)
        func_v = self._amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        return (func_m, func_v)

    def _compute_covariance(self, x1, x2=None):
        if x2 is None:
            return self._amp2 * (self._cov_func(self._ls, x1, None)
                                + 1e-6*np.eye(x1.shape[0]))
        else:
            return self._amp2 * self._cov_func(self._ls, x1, x2)
        
    def getGradients(self, xstar):
        xstar = np.array([xstar])
        # gradient of mean
        # dk(X,x*) / dx
        dk = self._amp2 * self._covar_derivative(self._ls, self._X, xstar)
        dk = np.squeeze(dk)
        #If dk is written like below, there's no need to squeeze.
        #But we stick to spear mint as close as possible.
        #Seems like the version below has a different sign!
        #dk = self._amp2 * self._covar_derivative(self._ls, xstar, self._X)[0]
        grad_m = np.dot(self._alpha.T, dk)
        
        # gradient of variance
        k = self._compute_covariance(self._X, xstar)
        #kK = k^T * K^-1
        kK = spla.cho_solve((self._L, True), k)
        #kK^Tdk=s'(x). So for the derivative of v(x) in terms of s(x) we have:
        #v(x)=s^2(x) <=> v'(x)=2s(x)*s'(x)
        grad_v = -2 * np.dot(kK.T, dk)
        #As in spear mint grad_v is of the form [[v1, v2, ...]]
        #TODO: Check if this is really necessary. Seems dirty.
        #TODO: Appearantly the sign of the mean gradient is wrong!
        return (-grad_m, grad_v)

    def getNoise(self):
        return self._noise
        
