'''
Created on 09.12.2013

@author: Aaron Klein, Simon Bartels
'''

import gp
import numpy as np
import scipy.optimize as spo
import scipy.stats    as sps
import util
from model.gp.gp_model import GPModel
from compiler.ast import IfExp
import scipy.linalg as spla
from model.gp import gp_model, gp_model_factory


class EntropySearchChooser(object):
    '''
    classdocs
    '''


    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20):
        '''
        Constructor
        '''

        self._mcmc_iters = mcmc_iters
        self._noiseless = noiseless
        self.burnin = burnin
        self.pending_samples = pending_samples
        self.covar_name = covar
        self.grid_subset = grid_subset
        self.expt_dir = expt_dir
        self._hyper_samples = []

        
        self._is_initialized = False
        # TODO: replace members by local variables
        #self.iter = number_of_iterations

    def _real_init(self, dims, values):
        self._is_initialized = True
        
        # Input dimensionality.
        self.D = dims

        # Initial length scales.
        self._ls = np.ones(self.D)

        # Initial amplitude.
        self._amp2 = np.std(values)+1e-4

        # Initial observation noise.
        self._noise = 1e-3

        # Initial mean.
        self._mean = np.mean(values)

        # Save hyperparameter samples
        self._hyper_samples.append((self._mean, self._noise, self._amp2,
                                       self._ls))


        self._cov_func = getattr(gp, self.covar_name)
        self._covar_derivative = getattr(gp, "grad_" + self.covar_name)

        self._noise_scale = 0.1  # horseshoe prior
        self._amp2_scale  = 1    # zero-mean log normal prior
        self._max_ls      = 10    # top-hat prior on length scales

    def next(self, grid, values, durations,
             candidates, pending, complete):
        '''
        Uses only ONE model!
        '''
        comp = grid[complete,:]
        if comp.shape[0] < 2:
            return candidates[0]
        cand = grid[candidates,:]
        cand = cand[0:10]
        
        vals = values[complete]
        numcand = cand.shape[0]
        dimension = comp.shape[1]
        
        #TODO: Workaround for hyperparameter sampling
        self._func_values = vals
        self._input_points = comp
        
        if self._is_initialized == False:
            self._real_init(dimension, vals)
        
        # optimization bounds
        b = []  
        for i in xrange(0, dimension):
            b.append((0, 1))
                    
        #Dimension of our input variable X
        dim = (-1, cand[0].shape[0])
       
        #Slice sampling of hyper parameters
        self._sampling_hyperparameters()
        
        ei_candidates = self._compute_candidates()
        
        
        for h in xrange(0, len(self._hyper_samples)-1):
            #TODO: copy hyperparameters to local values
            gp = GPModel(self._input_points, self._func_values, mean, noise, amp2, ls, covarname)
            (mu, sigma) = gp.predict( candidates, True)
            pmin = compute_pmin(ei_candidates, mu, sigma)
            for x in xrange(0, candidates.shape[0]):
                a_kl(x) = compute_entropy(gp, mu, sigma, pmin)

                    
        best_cand = np.argmax(np.mean(overall_ac_value, axis=1))
        
        return int(candidates[best_cand])


    def _sampling_hyperparameters(self):
      
        self._mean = np.mean(self._func_values)
        
        # sample hyper parameters
        for i in xrange(0, self._mcmc_iters ):
            if self._noiseless:
                self._noise = 1e-3
                self._sample_noiseless(self._input_points, self._func_values)
            else:
                self._sample_noisy(self._input_points, self._func_values)
            self._sample_ls(self._input_points, self._func_values)
            self._hyper_samples.append((self._mean, self._noise, self._amp2, self._ls))

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
                