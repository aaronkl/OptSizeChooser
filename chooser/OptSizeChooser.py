'''
Created on 18.11.2013

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
from acquisition_functions.expected_improvement import ExpectedImprovement
from acquisition_functions.entropy_search import EntropySearch
import support.Visualizer
from multiprocessing import Pool
from support.hyper_parameter_sampling import sample_hyperparameters

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return OptSizeChooser(expt_dir, **args)

def _iterate_over_candidates(gp, cand, vals, starting_point):
    '''
    Multiprocessing can only call functions that are pickable. Thus these method must be at the top level
    of a class. This is the workaround.
    Args:
        gp: a gaussian process
        cand: the set of candidate points
        vals: all values of the objective function that have been observed so far (NOT necessarily of the candidates!) 
        starting_point: a start point to maximize the acquisition function (for example the current best)
    Returns:
        an array of acquisition function values (for each candidate one value)
    '''
    #This is what we want to distribute over different processes
    #We'll iterate over all candidates for each GP in a different thread
    #Iterate over all candidates
    ac_func = _initialize_acquisition_function(starting_point, vals, gp)
    ac_values = np.zeros(len(cand))
    for i in xrange(0, len(cand)):
        ac_values[i] = ac_func.compute(cand[i], gp)
    return ac_values

def _initialize_acquisition_function(starting_point, vals, gp):
        #TODO: generalize initialisation of the acquisition function
        ei = ExpectedImprovement(vals)
        def log_proposal_measure(x):
            if np.any(x<0) or np.any(x>1):
                return -np.inf
            v = ei.compute(x, gp)
            return np.log(v+1e-10)
        return EntropySearch(log_proposal_measure, starting_point)

class OptSizeChooser():
    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20):
        '''
        Constructor
        '''
        self._mcmc_iters = mcmc_iters
        self._noiseless = noiseless
        self._burnin = burnin
        self.pending_samples = pending_samples
        self.grid_subset = grid_subset
        self.expt_dir = expt_dir
        self._hyper_samples = []
        self._cov_func = getattr(gp, covar)
        self._covar_derivative = getattr(gp, "grad_" + covar)
        self._is_initialized = False

    def _real_init(self, dims, values):
        '''
        Performs some more initialization with the first call of next()
        Args:
            dims: the dimension of the objective function
            values: the values that have been observed so far
        '''
        self._is_initialized = True

        # Initial length scales.
        ls = np.ones(dims)

        # Initial amplitude.
        amp2 = np.std(values)+1e-4

        # Initial observation noise.
        noise = 1e-3

        # Initial mean.
        mean = np.mean(values)

        # Save hyperparameter samples
        self._hyper_samples.append((mean, noise, amp2,
                                       ls))

    def next(self, grid, values, durations,
             candidates, pending, complete):
        #TODO: BURN IN!
        comp = grid[complete,:]
        if comp.shape[0] < 2:
            return candidates[0]
        cand = grid[candidates,:]
        #cand = cand[0:100]
        
        vals = values[complete]
        dimension = comp.shape[1]
        
        if not self._is_initialized:
            self._real_init(dimension, vals)
            #burn in
            (_, noise, amp2, ls) = self._hyper_samples[len(self._hyper_samples)-1]
            self._hyper_samples = sample_hyperparameters(self._burnin, self._noiseless, comp, vals, self._cov_func, noise, amp2, ls)
        
        # Spray a set of candidates around the min so far
#         best_comp = np.argmin(vals)
#         cand2 = np.vstack((np.random.randn(10,comp.shape[1])*0.001 +
#                            comp[best_comp,:], cand))

        #optimize around the local min
        #results_cand2 = [self._optimize_pt(c, b, incumbent) for c in cand2]
        
        #overall results of the acquisition functions for each candidate
        overall_ac_value = np.zeros(len(cand))
        
        #Slice sampling of hyper parameters
        #Get last sampled hyper-parameters
        (_, noise, amp2, ls) = self._hyper_samples[len(self._hyper_samples)-1]
        #TODO: remove
        print "last hyper parameters: " + str(self._hyper_samples[len(self._hyper_samples)-1])
        self._hyper_samples = sample_hyperparameters(self._mcmc_iters, self._noiseless, comp, vals, self._cov_func, noise, amp2, ls)
        
        #Prepare multiprocessing
        #TODO: remove
        print "going to use " + str(self.grid_subset)  + " worker threads"
        pool = Pool(self.grid_subset)
        results = []
        starting_point = comp[np.argmin(vals)] 
        #Create a GP for each hyper-parameter sample
        for h in range(0, len(self._hyper_samples)-1):
            hyper = self._hyper_samples[h]
            gp = GPModel(comp, vals, hyper[0], hyper[1], hyper[2], hyper[3])
            #Iterate over all candidates for each GP in parallel
            #apparently the last , needs to be there
            results.append(pool.apply_async(_iterate_over_candidates, (gp, cand, vals, starting_point,)))
        pool.close()
        pool.join()
        for res in results:
            res.wait()
            ac_vals = res.get()
            overall_ac_value+=ac_vals
            
        best_cand = np.argmax(overall_ac_value)

        return int(candidates[best_cand])
        