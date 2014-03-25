'''
Created on 18.11.2013

@author: Aaron Klein, Simon Bartels
'''

import gp
import numpy as np
import scipy.optimize as spo
import copy
import util
from util import slice_sample


from gp_model import GPModel
from acquisition_functions.expected_improvement import ExpectedImprovement
from acquisition_functions.entropy_search import EntropySearch
from acquisition_functions.entropy_search_big_data import EntropySearchBigData
from multiprocessing import Pool
from support.hyper_parameter_sampling import sample_hyperparameters
import traceback
from helpers import log
from gp_model import getNumberOfParameters, fetchKernel
import gp_model
from support.hyper_parameter_sampling import handle_slice_sampler_exception
import math

'''
The number of candidates if using local search.
Must be a factor of 2.
'''
NUMBER_OF_CANDIDATES = 10


def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return OptSizeChooser(expt_dir, **args)


class OptSizeChooser(object):

    def __init__(self, expt_dir, covar='Matern52', cost_covar='Polynomial3',
                 mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20,
                 pool_size=16):

        seed = np.random.randint(65000)
        log("using seed: " + str(seed))
        np.random.seed(seed)

        self.pending_samples = pending_samples
        self.grid_subset = grid_subset
        self.expt_dir = expt_dir
        self._hyper_samples = []
        self._cost_func_hyper_param = []
        self._is_initialized = False
        self._pool_size = int(pool_size)
        self._mcmc_iters = int(mcmc_iters)
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
        self._hyper_samples = sample_hyperparameters(self._burnin,
                                                     self._noiseless,
                                                     comp, values,
                                                     self._cov_func,
                                                     noise, amp2, ls)
        if self._model_costs:
            amp2 = np.std(durations) + 1e-4
            ls = np.ones(getNumberOfParameters(self._covar, dims))
            #burn in for the cost models
            self._cost_func_hyper_param = sample_hyperparameters(self._burnin,
                                                         self._noiseless,
                                                         comp, durations,
                                                         self._cost_cov_func,
                                                         noise,
                                                         amp2, ls)

        import Visualizer as vis
        self._visualizer = vis.Visualizer(comp.shape[0] - 2)

    def next(self, grid, values, durations, candidates, pending, complete):

        comp = grid[complete, :]
        if comp.shape[0] < 2:
            c = grid[candidates[0]]
            log("Evaluating: " + str(c))
            return candidates[0]

        vals = values[complete]
        dimension = comp.shape[1]
        durs = durations[complete]

        #TODO: remove (only for Deception)
        #####################################################################################
        #durs = 3 + (comp[:, 0] * 5 + 5 - 5 * comp[:, 1]) ** 3
        #####################################################################################

        #TODO: Adapt the code
        self._comp = comp
        self._vals = vals

        if not self._is_initialized:
            self._real_init(dimension, comp, vals, durs)

        #initialize Gaussian processes
        self._initialize_models(comp, vals, durs)
#         #####################################################################################
#         #TODO: remove (this is only for Branin...)
#         def branin(x):
#             x[0] = x[0]*15
#             x[1] = (x[1]*15)-5
#             y = np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0] - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10;
#             return y
#         branin_mins = np.array([[-np.pi, 12.275], [np.pi, 2.275], [3 * np.pi, 2.475]])
#         #branin_mins = np.array([[12.275,-np.pi], [ 2.275,np.pi], [ 2.475, 3 * np.pi]])
# 
#         branin_mins[:, 0] = branin_mins[:, 0] / 15
#         branin_mins[:, 1] = (branin_mins[:, 1] + 5) / 15
# 
#         es = EntropySearch(comp, vals, self._models[0])
#         es._func_sample_locations[0:3] = branin_mins
#         for i in range(0, 3):
#             es._log_proposal_vals[i] = es._log_proposal_measure(es._func_sample_locations[i])
#         mean, L = es._gp.getCholeskyForJointSample(es._func_sample_locations)
#         pmin = es._compute_pmin_bins_faster(mean, L)
#         for i in range(0, es._func_sample_locations.shape[0]):
#             print str(branin(es._func_sample_locations[i])) \
#                 + " is min with probability " + str(pmin[i])
        #####################################################################################

        cand = grid[candidates, :]

        #Sample some candidates based on EI / Costs
        selected_candidates = self._preselect_candidates(NUMBER_OF_CANDIDATES,
                                    cand, comp, vals)

        #FIXME: Instead of sampling take the candidates from the sobel sequence
        #selected_candidates = cand[:NUMBER_OF_CANDIDATES]

        overall_entropy = np.zeros(selected_candidates.shape[0])

        for i in xrange(0, len(self._models)):
            overall_entropy += self._entropy_search(selected_candidates, comp,
                                                    vals, self._models[i],
                                                    self._cost_models[i])
        overall_entropy /= len(self._models)

        #best_cand = np.argmin(overall_entropy)
        best_cand = np.argmax(overall_entropy)

        #self._visualizer.plot_projected_gp(comp, vals,
#                                    self._cost_models[0], True)

#         self._visualizer.plot(comp, vals, self._models[0],
#                               self._cost_models[0],
#                               selected_candidates)

#         self._visualizer.plot3D(comp, vals, self._models[0],
#                                self._cost_models[0],
#                                selected_candidates[best_cand],
#                                selected_candidates)

        log("Evaluating: " + str(selected_candidates[best_cand]))

        return (len(candidates) + 1, selected_candidates[best_cand])

    def _initialize_models(self, comp, vals, durs):
        '''
        Initializes the models of the objective function and if required the
        models for the cost functions.

        Arguments:
        comp: the points where the objective function has been evaluated so far
        vals: the corresponding observed values
        durs: the time it took to compute the values

        Returns:
            a tuple of two lists. The first list is a list of Gaussian process
            models for the objective function. The second list is empty if
            self._model_costs is false. Otherwise it is a list of
            Gaussian processes that model the costs for evaluating the
            objective function. In this case the lists are of equal length.
        '''
        #Slice sampling of hyper parameters
        #Get last sampled hyper-parameters
        (_, noise, amp2, ls) = self._hyper_samples[len(self._hyper_samples) - 1]

        log("last hyper parameters: " +
            str(self._hyper_samples[len(self._hyper_samples) - 1]))

        self._hyper_samples = sample_hyperparameters(self._mcmc_iters,
                                                     self._noiseless,
                                                     comp, vals,
                                                     self._cov_func, noise,
                                                     amp2, ls)

        if self._model_costs:
                (_, noise, amp2, ls) = self._cost_func_hyper_param[len(self._cost_func_hyper_param) - 1]

                self._cost_func_hyper_param = sample_hyperparameters(
                                                        self._mcmc_iters,
                                                        self._noiseless,
                                                        comp, durs,
                                                        self._cost_cov_func,
                                                        noise, amp2, ls)

        self._models = []
        self._cost_models = []

        for h in range(0, len(self._hyper_samples)):
            hyper = self._hyper_samples[h]
            gp = GPModel(comp, vals, hyper[0], hyper[1],
                         hyper[2], hyper[3], self._covar)

            self._models.append(gp)

            if self._model_costs:
                cost_hyper = self._cost_func_hyper_param[h]

                cost_gp = GPModel(comp, durs, cost_hyper[0], cost_hyper[1],
                                  cost_hyper[2], cost_hyper[3],
                                  self._cost_covar)

                self._cost_models.append(cost_gp)

    def _preselect_candidates(self, num_of_cands, cand, comp, vals):

        overall_ei_values = np.zeros(cand.shape[0])
        for m in self._models:
            ei = ExpectedImprovement(comp, vals, m)
            for i in xrange(0, cand.shape[0]):
                ei_value = ei.compute(cand[i])
                overall_ei_values[i] += ei_value

        best_cands_ind = np.argmax(overall_ei_values)

        starting_point = cand[best_cands_ind]
        selected_cands = np.zeros([num_of_cands, starting_point.shape[0]])
        chain_length = 20 * starting_point.shape[0]
        #TODO: burnin?
        for i in range(0, num_of_cands):
            #this for loop ensures better mixing
            for c in range(0, chain_length):
                try:
                    starting_point = slice_sample(starting_point,
                                                  self._log_proposal_measure_with_cost)
                    #FIXME:
#                     starting_point = slice_sample(starting_point,
#                               self._log_proposal_measure)
                except Exception as e:
                    starting_point = handle_slice_sampler_exception(e,
                                                    starting_point,
                                                    self._log_proposal_measure_with_cost)
                    #FIXME:
#                     starting_point = handle_slice_sampler_exception(e,
#                                                     starting_point,
#                                                     self._log_proposal_measure)
            selected_cands[i] = starting_point

        return selected_cands

    def _entropy_search(self, cand, comp, vals, model, cost_model):
        #FIXME:
        entropy_estimator = EntropySearchBigData(comp, vals, model, cost_model)
#        entropy_estimator = EntropySearch(comp, vals, model, cost_model)
        entropy = np.zeros(cand.shape[0])

        for i in xrange(0, cand.shape[0]):
            entropy[i] = entropy_estimator.compute(cand[i])
        return entropy

    def _log_proposal_measure_with_cost(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        ei_value = 0
        for m in self._models:
            ei = ExpectedImprovement(self._comp, self._vals, m)
            ei_value += ei.compute(x) / len(self._models)

        cost = 0
        for c in self._cost_models:
            cost += c.predict(np.array([x]))

        cost /= len(self._cost_models)

        return np.log((ei_value + 1e-50) / (cost + 1e-50))

    def _log_proposal_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        ei_value = 0
        for m in self._models:
            ei = ExpectedImprovement(self._comp, self._vals, m)
            ei_value += ei.compute(x) / len(self._models)

        return np.log(ei_value + 1e-50)

    def _compute_negative_gradient_over_hypers(self, x, comp, vals):
        '''
        Computes negative value and negative gradient of all acquisition
        functions in the list for one candidate. The purpose of this function
        is to be called with a minimizer.

        Arguments:
        x:  the candidate acquisition_functions: a list of INITIALIZED
            acquisition functions dimension: how to reshape the candidate

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

