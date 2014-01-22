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
from acquisition_functions.entropy_search_big_data import EntropySearchBigData
from multiprocessing import Pool
from support.hyper_parameter_sampling import sample_hyperparameters
import traceback
from helpers import log
from model.gp.gp_model import BigData, getNumberOfParameters, Polynomial3, grad_BigData

'''
The number of candidates if using local search.
'''
NUMBER_OF_CANDIDATES = 100

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return OptSizeChooser(expt_dir, **args)

def _iterate_over_candidates(ac_func, cand):
    '''
    Is called by _apply_acquisition_function_asynchronously. Multiprocessing can only call functions that 
    are pickable. Thus these method must be at the top level of a class. This is the workaround.
    Args:
        ac_func: the acquisition function
        cand: the set of candidate points
    Returns:
        an array of acquisition function values (for each candidate one value)
    '''
    try:
        #This is what we want to distribute over different processes
        #We'll iterate over all candidates for each GP in a different thread
        ac_values = np.zeros(len(cand))
        #Iterate over all candidates
        for i in xrange(0, len(cand)):
            ac_values[i] = ac_func.compute(cand[i])
        return ac_values
    #This is to make the multi-process debugging easier
    except Exception, e: 
        print traceback.format_exc()
        raise e
    
def _call_minimizer(cand, func, arguments, opt_bounds):
    '''
    This function is also desgined to be called in parallel. It calls a minmizer with the given argument.
    Args:
        cand: the starting point for the minimizer
        func: the function to be minimized
        arguments: for func
        opt_bounds: the optimization bounds
    Returns:
        a triple consisting of the point, the value and the gradients
        
    '''
    try:
        return spo.fmin_l_bfgs_b(func, cand.flatten(), args=arguments, bounds=opt_bounds, disp=0)
    except Exception, e: 
        print traceback.format_exc()
        raise e

class OptSizeChooser(object):
    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20):
        '''
        Constructor
        '''
        seed = np.random.randint(65000)
        log("using seed: " + str(seed))
        np.random.seed(seed)
        self._mcmc_iters = mcmc_iters
        self._noiseless = noiseless
        self._burnin = burnin
        self.pending_samples = pending_samples
        self.grid_subset = grid_subset
        self.expt_dir = expt_dir
        self._hyper_samples = []
        self._cost_function_hyper_parameter_samples = []
        #TODO: generalize
        self._covar = 'BigData'
        self._cov_func = BigData #getattr(gp, covar)
        self._covar_derivative = grad_BigData #getattr(gp, "grad_" + covar)
        self._cost_covar = 'Polynomial3'
        self._cost_cov_func = Polynomial3
        self._is_initialized = False
        #TODO: remove
        self._do_local_search = False
        #TODO: remove
        self._model_costs = False
        #TODO: generalize!
        #TODO: remove
        self._ac_func = EntropySearch
        #the acquisition function to preselect candidates before giving it to the real acquisition function
        #only used if do_local_search is true
        self._preselection_ac_func = ExpectedImprovement

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
        ls = np.ones(getNumberOfParameters(self._covar, dims))

        # Initial amplitude.
        amp2 = np.std(values)+1e-4

        # Initial observation noise.
        noise = 1e-3
        
        #burn in
        self._hyper_samples = sample_hyperparameters(self._burnin, self._noiseless, comp, values, self._cov_func, 
                                                     noise, amp2, ls)
        
        if self._model_costs:
            amp2 = np.std(durations)+1e-4
            ls = np.ones(getNumberOfParameters(self._covar, dims))
            #burn in for the cost models
            self._cost_function_hyper_parameter_samples = sample_hyperparameters(self._burnin, self._noiseless, 
                                                                                 comp, durations, 
                                                                                 self._cost_cov_func, noise, amp2, 
                                                                                 ls)

    def next(self, grid, values, durations, candidates, pending, complete):
        comp = grid[complete,:]
        #TODO: find good initialization procedure!
        if comp.shape[0] < 2:
            c = grid[candidates[0]]
            log("Evaluating: " + str(c))
            return candidates[0]
            #return (len(candidates)+1, c)
                
        vals = values[complete]
        dimension = comp.shape[1]
        durs = durations[complete]
        
        #TODO: remove
        for i in range(0, durs.shape[0]):
            durs[i] = ((comp[i][0]*10+1)**3)*1000
        
        if not self._is_initialized:
            self._real_init(dimension, comp, vals, durs)

        #initialize Gaussian processes
        (models, cost_models) = self._initialize_models(comp, vals, durs)
        import support.Visualizer as vis
        vis.plot2DFunction(lambda x: models[len(models)-1].predict(x))
        #vis.plot2DFunction(lambda x: cost_models[len(cost_models)-1].predict(x))
               
        cand = grid[candidates,:]
        if self._do_local_search:
            cand = _preselect_candidates(NUMBER_OF_CANDIDATES, cand, comp, vals, 
                                         models, cost_models, self._preselection_ac_func)
        #TODO: remove (or remove this comment)
        else:
            cand = cand[:NUMBER_OF_CANDIDATES]
        ac_funcs = _initialize_acquisition_functions(self._ac_func, comp, vals, models, cost_models)
        #overall results of the acquisition functions for each candidate over all models
        overall_ac_value = _apply_acquisition_function_asynchronously(ac_funcs, cand, self.grid_subset)
            
        best_cand = np.argmax(overall_ac_value)
        if self._do_local_search:
            log("Evaluating: " + str(cand[best_cand]))
            return (len(candidates)+1, cand[best_cand])
        else:
            log("Evaluating: " + str(cand[best_cand]))
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
            gp = GPModel(comp, vals, hyper[0], hyper[1], hyper[2], hyper[3], self._covar)
            models.append(gp)
            if self._model_costs:
                cost_hyper = self._cost_function_hyper_parameter_samples[h]
                cost_gp = GPModel(comp, durs, cost_hyper[0], cost_hyper[1], cost_hyper[2], cost_hyper[3], self._cost_covar)
                cost_models.append(cost_gp)
        return (models, cost_models)
    
def _initialize_acquisition_functions(ac_func, comp, vals, models, cost_models):
    '''
    Initializes an acquisition function for each model.
    Args:
        ac_func: an (UNINITIALIZED) acquisition function
        comp: the points where the objective function has been evaluated so far
        vals: the corresponding observed values
        models: a list of models
        cost_models: OPTIONAL. will only be used if the length of the list is equal to the length of the list of models
     Returns:
         a list of initialized acquisition functions
    '''
    model_costs = len(models) == len(cost_models)
    cost_model = None
    ac_funcs = []
    for i in range(0, len(models)):
        if model_costs:
            cost_model = cost_models[i]
        ac_funcs.append(ac_func(comp, vals, models[i], cost_model))
    return ac_funcs
    
def _apply_acquisition_function_asynchronously(ac_funcs, cand, pool_size=16):
    '''
    Applies the given acquisition function over all candidates for each model in parallel.
    Calls _iterate_over_candidates for that purpose.
    Args:
        ac_func: the acquisition function (will be initialized)
        models: a list of models
        cost_models: OPTIONAL. will only be used if the length of the list is equal to the length of the list of models
        cand: a list of candidates to iterate over
        pool_size: the number of worker threads to use
    Returns:
        a numpy vector of values, one entry for each entry in cand
    '''
    
    #Prepare multiprocessing
    log("going to use " + str(pool_size)  + " worker threads")
    pool = Pool(pool_size)
    results = []
    #Create a GP for each hyper-parameter sample
    for acf in ac_funcs:
        #Iterate over all candidates for each GP in parallel
        #apparently the last , needs to be there
        results.append(pool.apply_async(_iterate_over_candidates, (acf, cand,)))
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
    ac_funcs = _initialize_acquisition_functions(ac_func, comp, vals, models, cost_models)
    #add 10 random candidates around the current minimum
    cand = np.vstack((np.random.randn(10,comp.shape[1])*0.001 +
                           comp[np.argmin(vals),:], cand))
    overall_ac_value = _apply_acquisition_function_asynchronously(ac_funcs, cand)
    best_cands_indices = overall_ac_value.argsort()[-number_of_points_to_return:][::-1]
    best_cands = cand[best_cands_indices, :]
    locally_optimized = np.zeros(best_cands.shape)
    opt_bounds = []# optimization bounds
    for i in xrange(0, best_cands.shape[1]):
        opt_bounds.append((0, 1))
    dimension = (-1, cand.shape[0])
    
    pool_size = 16 #TODO: generalize
    log("going to use " + str(pool_size)  + " worker threads")
    pool = Pool(pool_size)
    #call minimizer in parallel
    results = [pool.apply_async(_call_minimizer, (best_cands[i], _compute_negative_gradient_over_hypers, 
                                                  (ac_funcs, dimension), opt_bounds)) 
               for i in range(0, best_cands.shape[0])]
    pool.close()
    pool.join()
    
    for i in range(0, best_cands.shape[0]):
        res = results[i]
        res.wait()
        try:
            p = res.get()[0]
            locally_optimized[i] = p
        except Exception, ex:
            log("Worker Thread reported exception:" + str(ex) + "\n Action: ignoring result")
    #remove duplicate entries and fill with best_cands    
    preselected_candidates = list(set(tuple(p) for p in locally_optimized))
    n = len(preselected_candidates)
    for i in range(0, number_of_points_to_return-n):
        preselected_candidates.append(best_cands[i])
    
    #TODO: maybe it would be good to have random points here!
    #return the number_of_points_to_return best candidates
    return np.array(preselected_candidates)
    
def _compute_negative_gradient_over_hypers(x, acquisition_functions, dimension):
    '''
    Computes negative value and negative gradient of all acquisition functions in the list for one candidate.
    The purpose of this function is to be called with a minimizer (and acquisition functions are usually maximized).
    Args:
        x: the candidate
        acquisition_functions: a list of INITIALIZED acquisition functions
        dimension: how to reshape the candidate
    Returns:
        a tuple (-sum acf(x), -sum grad_acf(x))
    '''
    grad_sum = np.zeros(x.shape).flatten()
    #x = np.reshape(x, dimension)
    val_sum = 0
    for acf in acquisition_functions:
        (val, grad) = acf.compute(x, True)
        val_sum+=val
        grad_sum+=grad.flatten()
    return (-val_sum, -grad_sum)