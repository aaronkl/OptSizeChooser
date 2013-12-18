'''
Created on Oct 28, 2013

@author: Simon Bartels
'''

import gp
import numpy as np
import scipy.linalg as spla
from ..model import Model
import copy



class GPModel(Model):

    def __init__(self, X, y, mean, noise, amp2, ls, covarname="Matern52", cholesky=None, alpha=None, cov_func = None, covar_derivative=None):
        #TODO: just a stub
        '''
            Constructor
            Args:
            X: The observed inputs.
            y: The corresponding observed values.
            mean: the mean value
            noise: the noise value
            amp: the amplitude
            ls: the length scales (numpy array of length equialent to the dimension of the input points)
            covarname: the name of the covariance function (see spearmint class gp)
            
            The following arguments are only used internally for the copy() method.
            cholesky: if already available the cholesky of the kernel matrix
            alpha: if cholesky is passed this one needs to be set, too. It's L/L/(y-mean).
        '''
        
        self._X = X
        self._y = y
        self._ls = ls
        self._amp2 = amp2
        self._mean = mean
        self._noise = noise
        if cholesky is None:
            self._cov_func = getattr(gp, covarname)
            self._covar_derivative = getattr(gp, "grad_" + covarname)
            self._compute_cholesky()
        else:
            self._cov_func = cov_func
            self._covar_derivative =covar_derivative
            self._L = cholesky
            self._alpha = alpha
        
        
    def _compute_cholesky(self):
        #the Cholesky of the correlation matrix
        K = self._compute_covariance(self._X) + self._noise * np.eye(self._X.shape[0])
        self._L = spla.cholesky(K, lower=True)
        self._alpha = spla.cho_solve((self._L, True), self._y - self._mean)
        
    def predict(self, Xstar, variance=False):
        kXstar = self._compute_covariance(self._X, Xstar)
        func_m = np.dot(kXstar.T, self._alpha) + self._mean
        if not variance:
            return func_m
        
        beta = spla.solve_triangular(self._L, kXstar, lower=True)
        func_v = self._amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        return (func_m, func_v)

    def predict_vector(self, input_point):
        (func_m, func_v) = self.predict(np.array([input_point]), True)
        return (func_m[0], func_v[0])
    
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
        #TODO: Apparently the sign of the mean gradient is wrong!
        return (-grad_m, grad_v)

    def getNoise(self):
        return self._noise
        
    def sample(self, x, omega):
        '''
            Gives a stochastic prediction at x
            Parameters In: omega = sample from standard normal distribution 
                            x = the prediction point (vector)
            Parameters Out: y = function value at x
        '''
        x = np.array([x])
        cholsolve = spla.cho_solve((self._L, True), self._compute_covariance(self._X, x))
        cholesky = np.sqrt(self._compute_covariance(x, x) - 
                           np.dot(self._compute_covariance(x, self._X), cholsolve))
        y = self.predict(x,False) + cholesky * omega
        #y is in the form [[value]] (1x1 matrix)
        return y[0][0]
        
    def update(self, x, y):
        '''
            Adds x,y to the observation and creates a new GP 
            Args:
                x: a numpy vector
                y: a single value
        '''
        self._X = np.append(self._X, np.array([x]), axis=0)
        self._y = np.append(self._y, np.array([y]), axis=0)
        #TODO: Use factor update
        self._compute_cholesky()
        
    def drawJointSample(self, Xstar, omega):
        '''
            Draws a joint sample at the given points.
            Args:
                X: a numpy array of points, i.e. a matrix
                omega: a vector of samples from the standard normal distribution, one for each point
            Returns:
                a numpy array (vector)
        '''
        kXstar = self._compute_covariance(self._X, Xstar)
        cholsolve = spla.cho_solve((self._L, True), kXstar)
        Sigma = (self._compute_covariance(Xstar, Xstar) -
                  np.dot(kXstar.T, cholsolve))
        cholesky = spla.cholesky(Sigma + 1e-6*np.eye(Sigma.shape[0]))
        y = self.predict(Xstar,False) + np.dot(cholesky, omega)
        return y
    
    def copy(self):
        '''
        Returns a copy of this object.
        Returns:
            a copy
        '''
        X = copy.copy(self._X)
        y = copy.copy(self._y)
        ls = copy.copy(self._ls)
        L = copy.copy(self._L)
        alpha = self._alpha
        return GPModel(X, y, self._mean, self._noise, self._amp2, ls, None, L, alpha, self._cov_func, self._covar_derivative)