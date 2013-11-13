'''
Created on 06.11.2013

@author: Aaron Klein
'''

import gp
import numpy as np
import scipy.linalg as spla
import scipy.weave
import matplotlib.pyplot as plt
import math
import util
from ..model import Model
import gp_model
from support.Visualizer import Visualizer



class GaussianProcessMixture(Model):
    '''
    classdocs
    '''


    def __init__(self, input_points, func_values, mean, noise, amp2, ls, covarname="Matern52"):
        '''
        Constructor
        '''
        # TODO: replace members by local variables
        #self.iter = number_of_iterations
        self._hyper_samples = []
        self._ls = ls
        self._amp2 = amp2
        self._cov_func = getattr(gp, covarname)
        self._covar_derivative = getattr(gp, "grad_" + covarname)
        self._mean = mean
        self._noise = noise
        self._input_points = input_points
        self._func_values = func_values
        self._iter = 10
        self._noiseless = True
        self._noise_scale = 0.1  # horseshoe prior
        self._amp2_scale  = 1    # zero-mean log normal prior
        self._max_ls      = 10    # top-hat prior on length scales

    def getNoise(self):
        return self._noise
     
    def sampling_hyperparameters(self):
        
        self._ls = np.ones(self._input_points.shape[1])
        self._mean = np.mean(self._func_values)
        
        # sample hyper parameters
        for i in xrange(0, self._iter ):
            if self._noiseless:
                self._noise = 1e-3
                self._sample_noiseless(self._input_points, self._func_values)
            else:
                self._sample_noisy(self._input_points, self._func_values)
            self._sample_ls(self._input_points, self._func_values)
            self._hyper_samples.append((self._mean, self._noise, self._amp2, self._ls))
            #TODO: remove
            print "hypers: mean, noise, amp, ls: " + str((self._mean, self._noise, self._amp2, self._ls))

#            visualizer = Visualizer()
#
#            gp = gp_model.GPModel(self._input_points, self._func_values, self._mean, self._noise, self._amp2, self._ls)    
#            visualizer.plot_projected_gp(self._input_points, self._func_values, gp)
            
    def getGradients(self, xstar):
        xstar = np.array([xstar])
        # gradient of mean
        # dk(X,x*) / dx

        # gradient of variance
        k = self._compute_covariance(self._input_points, xstar)
        #kK = k^T * K^-1
        self._K = self._compute_covariance(self._input_points) + self._noise * np.eye(self._input_points.shape[0])
        self._L = spla.cholesky(self._K, lower=True)
        self._alpha = spla.cho_solve((self._L, True), self._y - self._mean)


        dk = self._amp2 * self._covar_derivative(self._ls, self._input_points, xstar)
        dk = np.squeeze(dk)
        #If dk is written like below, there's no need to squeeze.
        #But we stick to spear mint as close as possible.
        #Seems like the version below has a different sign!
        #dk = self._amp2 * self._covar_derivative(self._ls, xstar, self._X)[0]
        grad_m = np.dot(self._alpha.T, dk)
        
        kK = spla.cho_solve((self._L, True), k)
        #kK^Tdk=s'(x). So for the derivative of v(x) in terms of s(x) we have:
        #v(x)=s^2(x) <=> v'(x)=2s(x)*s'(x)
        grad_v = -2 * np.dot(kK.T, dk)
        #As in spear mint grad_v is of the form [[v1, v2, ...]]
        #TODO: Check if this is really necessary. Seems dirty.
        #TODO: Appearantly the sign of the mean gradient is wrong!
        
        return (-grad_m, grad_v)
         
    def predict(self, test_points, isVariance=True):

        
        means = np.zeros(test_points.shape[0])        
        variances = np.zeros(test_points.shape[0])
        for k in xrange(0,test_points.shape[0]):
            
            test_point = test_points[k]
           
            mean_predictions = np.zeros( len(self._hyper_samples))
            covar_predictions = np.zeros( len(self._hyper_samples))
            
            for h in range(0, len(self._hyper_samples)-1):
                hyper = self._hyper_samples[h]
                gp = gp_model.GPModel(self._input_points, self._func_values, hyper[0], hyper[1], hyper[2], hyper[3])
                (m, c) = gp.predict(np.array([test_point]), True)
                
               
                mean_predictions[h] = m[0]
                
                covar_predictions[h] = c[0]
    
            # compute the mean of the predictive means of the gaussian processes
            #TODO: numerical instabel 
            mean = 0
            for i in range(0, mean_predictions.shape[0]):
                mean += mean_predictions[i]
                
            mean = mean / (len(self._hyper_samples) - 1)
            #print mean
            variance = 0
            for i in range(0, covar_predictions.shape[0]):
                variance += covar_predictions[i]
            
            variance = variance / (len(self._hyper_samples)-1)
            
            means[k] = mean
            variances[k] = variance
            
        return (means, variances)
   
    def _sample_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self._max_ls):
                return -np.inf

            cov = (self._amp2 * (self._cov_func(ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + self._noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self._mean)

            lp = (-np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - self._mean, solve))
            return lp

        self._ls = util.slice_sample(self._ls, logprob, compwise=True)

    def _sample_noisy(self, comp, vals):
        def logprob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            noise = hypers[2]

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf

            cov = (amp2 * (self._cov_func(self._ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self._noise_scale / noise) ** 2))

            # Roll in amplitude lognormal prior
            lp -= 0.5 * (np.log(amp2) / self._amp2_scale) ** 2

            return lp

        hypers = util.slice_sample(np.array(
                [self._mean, self._amp2, self._noise]), logprob, compwise=False)
        self._mean = hypers[0]
        self._amp2 = hypers[1]
        self._noise = hypers[2]

    def _sample_noiseless(self, comp, vals):
        def logprob(hypers):
            mean = hypers[0]
            amp2 = hypers[1]
            noise = 1e-3

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0:
                return -np.inf

            cov = (amp2 * (self._cov_func(self._ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5 * (np.log(amp2) / self._amp2_scale) ** 2

            return lp

        hypers = util.slice_sample(np.array(
                [self._mean, self._amp2, self._noise]), logprob, compwise=False)
        self._mean = hypers[0]
        self._amp2 = hypers[1]
        self._noise = 1e-3


