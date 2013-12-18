'''
Created on 18.11.2013

@author: Aaron Klein, Simon Bartels
'''

import gp
import numpy as np
import scipy.optimize as spo
import util
from model.gp.gp_model import GPModel
from acquisition_functions.expected_improvement import ExpectedImprovement
from acquisition_functions.entropy_search import EntropySearch
from multiprocessing import Pool
from support.hyper_parameter_sampling import sample_hyperparameters
import traceback
from helpers import log

'''
The number of candidates if using local search.
'''
NUMBER_OF_CANDIDATES = 100

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return OptSizeChooser(expt_dir, **args)

def _apply_acquisition_function_asynchronously(ac_func, models, cost_models, cand, comp, vals, pool_size=16):
    '''
    Applies the given acquisition function over all candidates for each model in parallel.
    Calls _iterate_over_candidates for that purpose.
    Multiprocessing can only call functions that are pickable. Thus these method must be at the top level 
    of a class. This is the workaround.
    Args:
        ac_func: the acquisition function (will be initialized)
        models: a list of models
        cost_models: OPTIONAL. will only be used if the length of the list is equal to the length of the list of models
        cand: a list of candidates to iterate over
        comp: a list of points where the objective function has been evaluated so far
        vals: the values that have been observed corresponding to comp
        pool_size: the number of worker threads to use
    Returns:
        a numpy vector of values, one entry for each entry in cand
    '''
    model_costs = len(models) == len(cost_models)
    
    #Prepare multiprocessing
    log("going to use " + str(pool_size)  + " worker threads")
    pool = Pool(pool_size)
    results = []
    cost_gp = None
    #Create a GP for each hyper-parameter sample
    for m in range(0, len(models)-1):
        if model_costs:
            cost_gp = cost_models[m]
        
        #Iterate over all candidates for each GP in parallel
        #apparently the last , needs to be there
        results.append(pool.apply_async(_iterate_over_candidates, (ac_func, models[m], cost_gp, cand, comp, vals,)))
    pool.close()
    pool.join()
    
    overall_ac_value = np.zeros(len(cand))   
    #get results
    for res in results:
        res.wait()
        try:
            ac_vals = res.get()
            overall_ac_value+=ac_vals
        except Exception, ex:
            log("Worker Thread reported exception:" + str(ex) + "\n Action: ignoring result")
    return overall_ac_value

def _iterate_over_candidates(ac_func, gp, cost_gp, cand, comp, vals):
    '''
    Is called by _apply_acquisition_function_asynchronously. Multiprocessing can only call functions that 
    are pickable. Thus these method must be at the top level of a class. This is the workaround.
    Args:
        ac_func: the acquisition func (will be initialized)
        gp: a gaussian process
        cand: the set of candidate points
        vals: all values of the objective function that have been observed so far (NOT necessarily of the 
        candidates!) 
        starting_point: a start point to maximize the acquisition function (for example the current best)
    Returns:
        an array of acquisition function values (for each candidate one value)
    '''
    try:
        #This is what we want to distribute over different processes
        #We'll iterate over all candidates for each GP in a different thread
        
        ac_func.initialize(comp, vals, gp, cost_gp)
        ac_values = np.zeros(len(cand))
        #Iterate over all candidates
        for i in xrange(0, len(cand)):
            ac_values[i] = ac_func.compute(cand[i])
        return ac_values
    #This is to make the multi-process debugging easier
    except Exception, e: 
        print traceback.format_exc()
        raise e
        

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
        self._cost_function_hyper_parameter_samples = []
        self._cov_func = getattr(gp, covar)
        self._covar_derivative = getattr(gp, "grad_" + covar)
        #TODO: set cost covariance function
        self._cost_cov_func = self._cov_func
        self._is_initialized = False
        self._do_local_search = True
        self._model_costs = True

    def _real_init(self, dims, comp, values, durations):
        '''
        Performs some more initialization with the first call of next(). Mainly slice sampling.
        Args:
            dims: the dimension of the objective function
            comp: the points that have been evaluated so far
            values: the values that have been observed so far
            durations: the time it took to compute the points
        '''
        self._is_initialized = True

        # Initial length scales.
        ls = np.ones(dims)

        # Initial amplitude.
        amp2 = np.std(values)+1e-4

        # Initial observation noise.
        noise = 1e-3
        
        #burn in
        self._hyper_samples = sample_hyperparameters(self._burnin, self._noiseless, comp, values, self._cov_func, 
                                                     noise, amp2, ls)
        
        if self._model_costs:
            amp2 = np.std(durations)+1e-4
            #burn in for the cost models
            self._cost_function_hyper_parameter_samples = sample_hyperparameters(self._burnin, self._noiseless, 
                                                                                 comp, durations, 
                                                                                 self._cost_cov_func, noise, amp2, 
                                                                                 ls)

        

    def next(self, grid, values, durations, candidates, pending, complete):
        comp = grid[complete,:]
        if comp.shape[0] < 2:
            return candidates[0]
                
        vals = values[complete]
        dimension = comp.shape[1]
        durs = durations[complete]
        
        if not self._is_initialized:
            self._real_init(dimension, comp, vals, durs)

        #initialize Gaussian processes
        (models, cost_models) = self._initialize_models(comp, vals, durs)
               
        cand = grid[candidates,:]
        if self._do_local_search:
            ei = ExpectedImprovement()
            cand = _preselect_candidates(NUMBER_OF_CANDIDATES, cand, comp, vals, models, cost_models, ei)
    
        #TODO: generalize!
        ac_func = EntropySearch()
        #overall results of the acquisition functions for each candidate over all models
        overall_ac_value = _apply_acquisition_function_asynchronously(ac_func, models, cost_models, cand, comp, 
                                                                      vals, self.grid_subset)
            
        best_cand = np.argmax(overall_ac_value)
        if self._do_local_search:
            log("Evaluating: " + str(cand[best_cand]))
            return (len(candidates)+1, cand[best_cand])
        else:
            log("Evaluating: " + str(candidates[best_cand]))
            return int(candidates[best_cand])
        
    def _initialize_models(self, comp, vals, durs):
        '''
        Initiales the models of the objective function and if required the models for the cost functions.
        Args:
            comp: the points where the objective function has been evaluated so far
            vals: the corresponding observed values
            durs: the time it took to compute the values
        Returns:
            a tuple of two lists. The first list is a list of Gaussian process models for the objective function.
            The second list is empty if self._model_costs is false. Otherwise it is a list of Gaussian processes
            that model the costs for evaluating the objective function. In this case the lists are of equal length.
        '''
        #Slice sampling of hyper parameters
        #Get last sampled hyper-parameters
        (_, noise, amp2, ls) = self._hyper_samples[len(self._hyper_samples)-1]
        log("last hyper parameters: " + str(self._hyper_samples[len(self._hyper_samples)-1]))
        self._hyper_samples = sample_hyperparameters(self._mcmc_iters, self._noiseless, comp, vals, 
                                                     self._cov_func, noise, amp2, ls)
        
        if self._model_costs:
                (_, noise, amp2, ls) = self._cost_function_hyper_parameter_samples[len(self._cost_function_hyper_parameter_samples)-1]
                self._cost_function_hyper_parameter_samples = sample_hyperparameters(self._mcmc_iters, 
                                                                                     self._noiseless, comp, durs, 
                                                                                     self._cost_cov_func, noise, 
                                                                                     amp2, ls)

        models = []
        cost_models = []
        for h in range(0, len(self._hyper_samples)-1):
            hyper = self._hyper_samples[h]
            gp = GPModel(comp, vals, hyper[0], hyper[1], hyper[2], hyper[3])
            models.append(gp)
            if self._model_costs:
                cost_hyper = self._cost_function_hyper_parameter_samples[h]
                cost_gp = GPModel(comp, durs, cost_hyper[0], cost_hyper[1], cost_hyper[2], cost_hyper[3])
                cost_models.append(cost_gp)
        return (models, cost_models)
                
def _preselect_candidates(number_of_points_to_return, cand, comp, vals, models, cost_models, ac_func):
    '''
    Evaluates the acquisition function for all candidates over all models. Then selects the number_of_points_to_return
    best and optimizes them locally. (So the acquisition function MUST be capable of computing gradients)
    Args:
        number_of_points_to_return: the number of candidates that are to be returned
        cand: the available candidates
        comp: the points where the objective function has been evaluated so far
        vals: the corresponding observed values
        models: a list of Gaussian processes that model the objective function
        cost_models: OPTIONAL. will only be used if the length of the list is equal to the length of the list of models
        ac_func: an acquisition function that is capable of computing gradients (will be initialized)
    Returns:
        A numpy matrix consisting of the number_of_points_to_return best points 
        
    '''
    overall_ac_value = _apply_acquisition_function_asynchronously(ac_func, models, cost_models, cand, comp, vals)
    #TODO: use local optimization
    #np.random.randn(10,comp.shape[1])*0.001 + current_best
    best_cands = cand[overall_ac_value.argsort()[-number_of_points_to_return:][::-1], :]

#     locally_optimized = np.zeros(best_cands.shape)
#     opt_bounds = []# optimization bounds
#     for i in xrange(0, best_cands.shape[1]):
#         opt_bounds.append((0, 1))
#         
#     dimension = (-1, best_cands.shape[1])
#     #TODO: maybe parallelize
#     for i in range(0, best_cands.shape[0]):
#         locally_optimized[i] = spo.fmin_l_bfgs_b(_optimize_over_hypers,
#                             best_cands[i].flatten(), args=(ac_func, models, cost_models, comp, vals, dimension),
#                             bounds=opt_bounds, disp=0)[0]
                            
    #return the NUMBER_OF_CANDIDATES best candidates
    #TODO: return locally_optimized
    return best_cands

def _optimize_over_hypers(x, acquisition_function, models, cost_models, comp, vals, dimension):
    #FIXME: broken
    x = x[0] #for which reason ever, x is is of the form [vector]
    grad_sum = np.zeros(x.shape).flatten()
    x = np.reshape(x, dimension)
    use_cost_models = len(cost_models) == len(models)
    cost_model = None
    val_sum = 0
    for i in range(0, len(models)-1):
        if use_cost_models:
            cost_model = cost_models[i]
        acquisition_function.initialize(comp, vals, models[i], cost_model)
        (val, grad) = acquisition_function.compute(x, True)
        val_sum+=val
        grad_sum+=grad
    return (val_sum, grad_sum.flatten())