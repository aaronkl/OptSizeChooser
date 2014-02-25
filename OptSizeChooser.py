'''
Created on 18.11.2013

@author: Aaron Klein, Simon Bartels
'''

import gp
import numpy as np
import scipy.optimize as spo
import util
import copy

from gp_model import GPModel
from acquisition_functions.expected_improvement import ExpectedImprovement
from acquisition_functions.entropy_search import EntropySearch
from acquisition_functions.entropy_search_big_data import EntropySearchBigData
from multiprocessing import Pool
from support.hyper_parameter_sampling import sample_hyperparameters
import traceback
from helpers import log
from gp_model import getNumberOfParameters, fetchKernel
from spearmint.chooser.OptSizeChooser import gp_model

'''
The number of candidates if using local search.
Must be a factor of 2.
'''
NUMBER_OF_CANDIDATES = 100


def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return OptSizeChooser(expt_dir, **args)


class OptSizeChooser(object):

    def __init__(self, expt_dir, covar='Matern52', cost_covar='Matern52',
                 mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20,
                 pool_size=16):
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

        self._mcmc_iters = mcmc_iters

        self._noiseless = noiseless
        self._burnin = burnin
        self._covar = covar
        self._cov_func, self._covar_derivative = fetchKernel(covar)
        self._cost_covar = cost_covar
        self._cost_cov_func, _ = fetchKernel(cost_covar)
        self._do_local_search = True
        self._model_costs = True

    def _real_init(self, dims, comp, values, durations):
        '''
        Performs some more initialization with the first call of next(). 
        Burn in of the hyper parameters
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
        amp2 = np.std(values) + 1e-4

        # Initial observation noise.
        noise = 1e-3

        #burn in
        self._hyper_samples = sample_hyperparameters(self._burnin, self._noiseless,
                                                     comp, values, self._cov_func,
                                                     noise, amp2, ls)
        if self._model_costs:
            amp2 = np.std(durations) + 1e-4
            ls = np.ones(getNumberOfParameters(self._covar, dims))
            #burn in for the cost models
            self._cost_function_hyper_parameter_samples = sample_hyperparameters(self._burnin, self._noiseless, 
                                                                                 comp, durations, 
                                                                                 self._cost_cov_func, noise, amp2, 
                                                                                 ls)

#         import support.Visualizer as vis
#         self._visualizer = vis.Visualizer(comp.shape[0] - 2)

    def next(self, grid, values, durations, candidates, pending, complete):

        comp = grid[complete, :]
        if comp.shape[0] < 2:
            c = grid[candidates[0]]
            log("Evaluating: " + str(c))
            return candidates[0]

        vals = values[complete]
        dimension = comp.shape[1]
        durs = durations[complete]

        if not self._is_initialized:
            self._real_init(dimension, comp, vals, durs)

        #initialize Gaussian processes
        self._initialize_models(comp, vals, durs)

        cand = grid[candidates, :]

        #Plotting
#         if cand.shape[1] == 1:
#             self._visualizer.plot_gp(comp, vals, self._models[0], is_cost=False)
#             self._visualizer.plot_gp(comp, vals, self._cost_models[0], is_cost=True)
#         elif cand.shape[1] == 2:
#             self._visualizer.plot_projected_gp(comp, vals, self._models[0], is_cost=False)
#             self._visualizer.plot_projected_gp(comp, vals, self._cost_models[0], is_cost=True)

        #self._visualizer.plot_two_dim_gp(comp, vals, self._models[0], is_cost=False)

        #Select best candidates and optimize them locally based on EI
        selected_candidates = self._preselect_candidates(NUMBER_OF_CANDIDATES,
                                    cand, comp, vals)

        #log("Employing " + str(self._pool_size)  + " threads to compute acquisition function value for "
        #    + str(cand.shape[0]) + "candidates.")
        #pool = Pool(self._pool_size)
        #overall_ac_value = [pool.apply_async(self._entropy_search,
        #                                     args=(self, selected_candidates, h))
        #                                     for h in self._hyper_samples]
        #pool.close()
        #pool.join()

        #Surface plot of the entropy
#         self._visualizer.plot_entropy_surface(comp, vals, self._models[0], self._cost_models[0])

        #Compute entropy for each gaussian process sample
        overall_entropy = np.zeros(selected_candidates.shape[0])
        for i in xrange(0, len(self._models)):
            overall_entropy += self._entropy_search(selected_candidates, comp,
                                                    vals, self._models[i],
                                                    self._cost_models[i])

        #if cand.shape[1] == 1:
        #    self._visualizer.plot_entropy_one_dim(selected_candidates, overall_entropy)
        #elif cand.shape[1] == 2:
        #    self._visualizer.plot_entropy_two_dim(selected_candidates, overall_entropy)

        best_cand = np.argmax(overall_entropy)

        log("Evaluating: " + str(selected_candidates[best_cand]))
        return (len(candidates) + 1, selected_candidates[best_cand])

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

        self._models = []
        self._cost_models = []
        for h in range(0, len(self._hyper_samples)-1):
            hyper = self._hyper_samples[h]
            gp = GPModel(comp, vals, hyper[0], hyper[1], hyper[2], hyper[3], self._covar)
            self._models.append(gp)
            if self._model_costs:
                cost_hyper = self._cost_function_hyper_parameter_samples[h]
                cost_gp = GPModel(comp, durs, cost_hyper[0], cost_hyper[1], cost_hyper[2], cost_hyper[3], self._cost_covar)
                self._cost_models.append(cost_gp)

    def _preselect_candidates(self, number_of_points_to_return, cand, comp, vals):
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

        #ac_funcs = self._initialize_acquisition_functions(ac_func, comp, vals)

        #add 10 random candidates around the current minimum
        cand = np.vstack((np.random.randn(10, comp.shape[1]) * 0.001 +
                               comp[np.argmin(vals), :], cand))

        overall_ei_values = np.zeros(cand.shape[0])
        for i in xrange(0, cand.shape[0]):
            for m in self._models:
                ei = ExpectedImprovement(comp, vals, m)
                ei_value = ei.compute(cand[i])
                overall_ei_values[i] += ei_value

        #overall_ac_value = _apply_acquisition_function_asynchronously(ac_funcs, cand, pool_size)

        best_cands_indices = overall_ei_values.argsort()[-number_of_points_to_return / 2:][::-1]
        best_cands = cand[best_cands_indices, :]
        locally_optimized = np.zeros([best_cands.shape[0], best_cands.shape[1]])
        # optimization bounds
        opt_bounds = []
        for i in xrange(0, best_cands.shape[1]):
            opt_bounds.append((0, 1))
        dimension = (-1, cand.shape[0])

        #log("Employing " + str(self._pool_size)  + " threads for local optimization of candidates.")
        #pool = Pool(self._pool_size)
        #call minimizer in parallel
        #results = [pool.apply_async(_call_minimizer, (best_cands[i], _compute_negative_gradient_over_hypers, 
        #                                              (comp, vals, dimension), opt_bounds)) 
        #           for i in range(0, best_cands.shape[0])]
        #pool.close()
        #pool.join()
#         pool = Pool(self._pool_size)
#         results = [pool.apply_async(optimize_pt,args=(
#                     c, opt_bounds, comp, vals, copy.copy(self))) for c in best_cands]
#         for res in results:
#             cand = np.vstack((cand, res.get(1e8)))
#         pool.close()
#         for i in range(0, best_cands.shape[0]):
#             res = results[i]
#             res.wait()
#             try:
#                 p = res.get()[0]
#                 locally_optimized[i] = p
#             except Exception, ex:
#                 log("Worker Thread reported exception:" + str(ex) + "\n Action: ignoring result")

        #TODO: Multithreading yes or no?
        for i in xrange(0, best_cands.shape[0]):
            res = spo.fmin_l_bfgs_b( self._compute_negative_gradient_over_hypers,
                                        best_cands[i], args=(comp, vals),
                                        bounds=opt_bounds, disp=0)
            locally_optimized[i] = res[0]

        #remove duplicate entries and fill with best_cands
        preselected_candidates = list(set(tuple(p) for p in locally_optimized))
        n = len(preselected_candidates)
        for i in range(0, number_of_points_to_return / 2 - n):
            preselected_candidates.append(best_cands[i])

        for i in range(0, number_of_points_to_return / 2):
            preselected_candidates.append(cand[i])

#         if cand.shape[1] == 1:
#             self._visualizer.plot_expected_improvement_one_dim(cand,
#                                         overall_ei_values,
#                                         best_cands,
#                                         np.array(preselected_candidates))
#         elif cand.shape[1] == 2:
#             self._visualizer.plot_expected_improvement_two_dim(cand,
#                                         overall_ei_values,
#                                         best_cands,
#                                         np.array(preselected_candidates))

        return np.array(preselected_candidates)

    def _compute_negative_gradient_over_hypers(self, x, comp, vals):
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

        val_sum = 0

        for m in self._models:
            ei = ExpectedImprovement(comp, vals, m)
            ei_value, ei_grad = ei.compute(x, True)
            val_sum += ei_value
            grad_sum += ei_grad.flatten()
        return (-val_sum, -grad_sum)

    def _entropy_search(self, cand, comp, vals, model, cost_model):

        entropy_estimator = EntropySearchBigData(comp, vals, model, cost_model)
        entropy = np.zeros(cand.shape[0])
        for i in xrange(0, cand.shape[0]):
            entropy[i] = entropy_estimator.compute(cand[i])

        return entropy
