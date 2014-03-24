'''
Created on 18.11.2013

@author: Aaron Klein, Simon Bartels
'''
import numpy as np
import scipy.optimize as spo
import util
from gp_model import GPModel
from acquisition_functions.expected_improvement import ExpectedImprovement
from acquisition_functions.entropy_search import EntropySearch
from acquisition_functions.entropy_search_big_data import EntropySearchBigData
from multiprocessing import Pool
from support.hyper_parameter_sampling import sample_hyperparameters, sample_from_proposal_measure
import traceback
from helpers import log
from gp_model import getNumberOfParameters, fetchKernel
from support.Visualizer import Visualizer

'''
The number of candidates if using local search.
Must be a factor of 2.
'''
NUMBER_OF_CANDIDATES = 200

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return OptSizeChooser(expt_dir, **args)

def _apply_acquisition_function_asynchronously(ac_funcs, cand, pool_size):
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
    log("Employing " + str(pool_size)  + " threads to compute acquisition function value for "
        + str(cand.shape[0]) + "candidates.")
    pool = Pool(pool_size)
    results = []
    #Create a GP for each hyper-parameter sample
    for acf in ac_funcs:
        #Iterate over all candidates for each GP in parallel
        #apparently the last , needs to be there
        results.append(pool.apply_async(_iterate_over_candidates, (acf, cand,)))
    pool.close()
    pool.join()

    number_of_acquisition_functions = len(ac_funcs)
    overall_ac_value = np.zeros(len(cand))
    #get results
    for res in results:
        res.wait()
        try:
            ac_vals = res.get()
            overall_ac_value += ac_vals/number_of_acquisition_functions
        except Exception, ex:
            log("Worker Thread reported exception:" + str(ex) + "\n Action: ignoring result")
    return overall_ac_value

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
    normalization_constant = len(acquisition_functions)
    grad_sum = np.zeros(x.shape).flatten()
    #x = np.reshape(x, dimension)
    val_sum = 0
    for acf in acquisition_functions:
        (val, grad) = acf.compute(x, True)
        val_sum+=val/normalization_constant
        grad_sum+=grad.flatten()/normalization_constant
    return (-val_sum, -grad_sum)

class OptSizeChooser(object):
    def __init__(self, expt_dir, covar='Matern52', cost_covar='Polynomial3', mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20, acquisition_function='EntropySearchBigData', 
                 model_costs=True, do_local_search=True, 
                 local_search_acquisition_function='ExpectedImprovement',
                 pool_size=16, do_visualization=False):
        #TODO: use arguments! (acquisition functions)
        '''
        Constructor
        '''
        seed = np.random.randint(65000)
        log("using seed: " + str(seed))
        np.random.seed(seed)
        self.pending_samples = pending_samples
        self.grid_subset = grid_subset
        self.expt_dir = expt_dir
        self._hyper_samples = []
        self._cost_function_hyper_parameter_samples = []
        self._is_initialized = False

        self._pool_size = int(pool_size)
        self._mcmc_iters = int(mcmc_iters)
        self._noiseless = bool(noiseless)
        self._burnin = int(burnin)
        self._covar = covar
        self._cov_func, self._covar_derivative = fetchKernel(covar)
        self._cost_covar = cost_covar
        self._cost_cov_func, _ = fetchKernel(cost_covar)
        self._do_local_search = True
        self._model_costs = True
        #TODO: if false check that acquisition function can handle that
        self._ac_func = EntropySearchBigData
        #the acquisition function to preselect candidates before giving it to the real acquisition function
        #only used if do_local_search is true
        #TODO: check that acquisition function computes gradients
        self._preselection_ac_func = ExpectedImprovement

        self._do_visualization = do_visualization
        if do_visualization:
            vis = Visualizer(0)
            self._visualize = vis.plot3D

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
        if comp.shape[0] < 2:
            c = grid[candidates[0]]
            log("Evaluating: " + str(c))
            return candidates[0]
            #return (len(candidates)+1, c)
                
        vals = values[complete]
        dimension = comp.shape[1]
        durs = durations[complete]

        
        if not self._is_initialized:
            self._real_init(dimension, comp, vals, durs)

        #initialize Gaussian processes
        (models, cost_models) = self._initialize_models(comp, vals, durs)

        cand = grid[candidates,:]
        if self._do_local_search:
            cand = self._preselect_candidates(NUMBER_OF_CANDIDATES, cand, comp, vals,
                                         models, cost_models, self._preselection_ac_func, self._pool_size)

        ac_funcs = self._initialize_acquisition_functions(self._ac_func, comp, vals, models, cost_models)
        #overall results of the acquisition functions for each candidate over all models
        overall_ac_value = _apply_acquisition_function_asynchronously(ac_funcs, cand, self._pool_size)
            
        best_cand = np.argmax(overall_ac_value)

        #do visualization
        if self._do_visualization:
            log('Visualizing...')
            self._visualize(comp, vals, models[0],
                           cost_models[0],
                           cand[best_cand],
                           cand)

        if self._do_local_search:
            log("Evaluating: " + str(cand[best_cand]))
            return (len(candidates)+1, cand[best_cand])
        else:
            log("Evaluating: " + str(cand[best_cand]))
            return int(candidates[best_cand])
        
    def _initialize_models(self, comp, vals, durs):
        '''
        Initializes the models of the objective function and if required the models for the cost functions.
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
        for h in range(0, len(self._hyper_samples)):
            hyper = self._hyper_samples[h]
            gp = GPModel(comp, vals, hyper[0], hyper[1], hyper[2], hyper[3], self._covar)
            models.append(gp)
            if self._model_costs:
                cost_hyper = self._cost_function_hyper_parameter_samples[h]
                cost_gp = GPModel(comp, durs, cost_hyper[0], cost_hyper[1], cost_hyper[2], cost_hyper[3], self._cost_covar)
                cost_models.append(cost_gp)
        return (models, cost_models)
    
    def _initialize_acquisition_functions(self, ac_func, comp, vals, models, cost_models):
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
        cost_model = None
        ac_funcs = []
        for i in range(0, len(models)):
            if self._model_costs:
                cost_model = cost_models[i]
            ac_funcs.append(ac_func(comp, vals, models[i], cost_model))
        return ac_funcs
                
    def _preselect_candidates(self, number_of_points_to_return, cand, comp, vals, models, cost_models, ac_func, pool_size):
        '''
        Evaluates the acquisition function for all candidates over all models. Then selects the
        number_of_points_to_return/2 best and optimizes them locally.
        (So the acquisition function MUST be capable of computing gradients)
        This will be the first half of the array.
        The other half is filled with the first number_of_points_to_return/2 entries in cand.
        Args:
            number_of_points_to_return: the number of candidates that are to be returned
            cand: the available candidates
            comp: the points where the objective function has been evaluated so far
            vals: the corresponding observed values
            models: a list of Gaussian processes that model the objective function
            cost_models: OPTIONAL. will only be used if the length of the list is equal to the length of the list of models
            ac_func: an acquisition function that is capable of computing gradients (will be initialized)
            pool_size: the number of threads to use
        Returns:
            A numpy matrix consisting of the number_of_points_to_return best points

        '''
        preselected_candidates = np.zeros([number_of_points_to_return, comp.shape[1]])
        ac_funcs = self._initialize_acquisition_functions(ac_func, comp, vals, models, cost_models)
        #add 10 random candidates around the current minimum
        cand = np.vstack((np.random.randn(10,comp.shape[1])*0.001 +
                               comp[np.argmin(vals),:], cand))
        overall_ac_value = _apply_acquisition_function_asynchronously(ac_funcs, cand, pool_size)
        best_cands_indices = overall_ac_value.argsort()[-number_of_points_to_return/2:][::-1]
        best_cands = cand[best_cands_indices, :]
        locally_optimized = np.zeros(best_cands.shape)
        opt_bounds = []# optimization bounds
        for i in xrange(0, best_cands.shape[1]):
            opt_bounds.append((0, 1))
        dimension = (-1, cand.shape[0])

        log("Employing " + str(pool_size)  + " threads for local optimization of candidates.")
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
        #remove duplicate entries
        locally_optimized = list(set(tuple(p) for p in locally_optimized))
        n = len(locally_optimized)
        preselected_candidates[0:n] = np.array(locally_optimized)

        #and substitute with samples
        number_of_acquisition_functions = len(ac_funcs)
        #the measure is the sum over all acquisition functions
        log_proposal_measure = lambda x: np.log(np.sum(np.array([ac_funcs[i].compute(x)/number_of_acquisition_functions
                                                                 for i in range(0, number_of_acquisition_functions)])))
        #for i in range(0, number_of_points_to_return/2-n):
            #preselected_candidates.append(best_cands[i])
        preselected_candidates[n:number_of_points_to_return/2-n] = sample_from_proposal_measure(
            preselected_candidates[0], log_proposal_measure, number_of_points_to_return/2-n)

        #add candidates from the Sobol sequence
        preselected_candidates[number_of_points_to_return/2:number_of_points_to_return] = cand[0:number_of_points_to_return/2]

        #return the number_of_points_to_return best candidates
        return preselected_candidates