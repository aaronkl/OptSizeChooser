'''
Created on 06.11.2013

@author: Aaron Klein
'''

import numpy as np
import scipy.linalg as spla
import scipy.weave
import matplotlib.pyplot as plt
import math
import util
from model.model import Model
from chooser.model.gp import gp_model


class GaussianProcessMixture(Model):
    '''
    classdocs
    '''


    def __init__(self, number_of_iterations):
        '''
        Constructor
        '''
        # TODO: replace members by local variables
        self.iter = number_of_iterations
        self.hyper_samples = []
        self._ls = 0
        self._amp2 = 1.0
        self._cov_func = globals()['ARDSE']
        self._mean = 0
        self._noise = 1e-3
        
    def sampling_hyperparameters(self, completed, func_values):
        
        self._ls = np.ones(completed.shape[1])
        self._mean = np.mean(func_values)
        
        # sample hyper parameters
        for i in xrange(self.iter):
            if self.noiseless:
                self.noise = 1e-3
                self._sample_noiseless(completed, func_values)
            else:
                self._sample_noisy(completed, func_values)
            self._sample_ls(completed, completed)
            self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))

         
    def predict_over_hyper_parameters(self, input_point, completed, func_values):
        
        mean_predictions = np.zeros((input_point.shape[0], len(self._hyper_samples)))
        covar_predictions = np.zeros((input_point.shape[0], len(self._hyper_samples)))
        
            
        for h in range(0, len(self._hyper_samples)):
            hyper = self._hyper_samples[h]
            gp = gp_model(completed, func_values, hyper[0], hyper[1], hyper[2], hyper[3], 'ARDSE')
            (m, c) = gp.predict(input_point)
            mean_predictions[:, h] = m
            covar_predictions[:, h] = c

            # compute the mean of the predictive means of the gaussian processes
        mean = np.zeros((100, 1))
        for i in range(0, mean_predictions.shape[0]):
            mean[i] = mean_predictions[i, :].mean()

        variance = np.zeros((100, 1))
        for i in range(0, covar_predictions.shape[0]):
            variance[i] = covar_predictions[i, :].mean()
            
        return (mean, variance)
   
    def _sample_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf

            cov = (self.amp2 * (self.cov_func(ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + self.noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp = (-np.sum(np.log(np.diag(chol))) - 
                      0.5 * np.dot(vals - self.mean, solve))
            return lp

        self.ls = util.slice_sample(self.ls, logprob, compwise=True)

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

            cov = (amp2 * (self.cov_func(self.ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale / noise) ** 2))

            # Roll in amplitude lognormal prior
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2

            return lp

        hypers = util.slice_sample(np.array(
                [self.mean, self.amp2, self.noise]), logprob, compwise=False)
        self.mean = hypers[0]
        self.amp2 = hypers[1]
        self.noise = hypers[2]

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

            cov = (amp2 * (self.cov_func(self.ls, comp, None) + 
                1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
            chol = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5 * (np.log(amp2) / self.amp2_scale) ** 2

            return lp

        hypers = util.slice_sample(np.array(
                [self.mean, self.amp2, self.noise]), logprob, compwise=False)
        self.mean = hypers[0]
        self.amp2 = hypers[1]
        self.noise = 1e-3


    '''
        Plots the dimension defined by parameter "index"
    '''
    def plot_one_dimension(self, X, y, index, default):
        
        plt.plot(X[:, index], y, 'g+')

        x = np.linspace(0, 10, 100)[:, np.newaxis]
        test_inputs = np.ones((100, 2)) * default
        for i in range(0, 100):
            test_inputs[i, index] = x[i]

        (mean, variance) = self.predict_over_hyper_parameters(test_inputs)

        lower_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            lower_bound[i] = mean[i] - math.sqrt(variance[i])         
        
        upper_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            upper_bound[i] = mean[i] + math.sqrt(variance[i])
        
        plt.plot(x, mean, 'b')
        plt.fill_between(test_inputs[:, 0], upper_bound[:, 0], lower_bound[:, 0], facecolor='red')
        plt.grid(True)

        plt.show()