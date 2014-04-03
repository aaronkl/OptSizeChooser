'''
Created on 18.11.2013

@author: Aaron Klein, Simon Bartels
'''

import numpy as np
import numpy.random as npr
import scipy.optimize as spo

from spearmint.util import unpack_args
from checkbox.properties import Time
from multiprocessing import Pool
from spearmint.helpers import log

from gp_model import GPModel
from gp_model import fetchKernel
from gp_model import getNumberOfParameters
from entropy import Entropy
from entropy_with_costs import EntropyWithCosts
from hyper_parameter_sampling import sample_hyperparameters
from support import compute_pmin_bins, sample_from_proposal_measure


def init(expt_dir, arg_string):
    args = unpack_args(arg_string)
    return EntropySearchChooser(expt_dir, **args)


class EntropySearchChooser(object):

    def __init__(self, expt_dir, covar='Matern52', cost_covar='Polynomial3',
                 mcmc_iters=10,
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20,
                 num_of_cands=100,
                 pool_size=16,
                 incumbent_inter_sample_distance=20,
                 incumbent_number_of_minima=10,
                 number_of_pmin_samples=1000):

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

        self._num_of_candidates = num_of_cands

        self._incumbent_inter_sample_distance = int(incumbent_inter_sample_distance)
        self._incumbent_number_of_minima = int(incumbent_number_of_minima)
        self._number_of_pmin_samples = number_of_pmin_samples

    def _real_init(self, dims, comp, values, durations):

        self._is_initialized = True

        # Initial length scales.
        ls = np.ones(getNumberOfParameters(self._covar, dims))

        # Initial amplitude.
        amp2 = np.std(values) + 1e-4

        # Initial observation noise.
        noise = 1e-3

        self._comp = comp
        self._vals = values

        #burn in
        self._hyper_samples = sample_hyperparameters(self._burnin, self._noiseless, self._comp, values,
                                                     self._cov_func, noise, amp2, ls)

        amp2 = np.std(durations) + 1e-4
        ls = np.ones(getNumberOfParameters(self._covar, dims))
        #burn in for the cost models
        self._cost_func_hyper_param = sample_hyperparameters(self._burnin, self._noiseless,
                                                     self._comp, durations, self._cost_cov_func,
                                                     noise, amp2, ls)

        #Flags
        self._withCosts = False
        self._withPlotting = False
        self._withPlotting3D = False

        if(self._withPlotting or self._withPlotting3D):
            import Visualizer as vis
            self._visualizer = vis.Visualizer(comp.shape[0] - 2)

    def next(self, grid, values, durations, candidates, pending, complete):

        self._comp = grid[complete, :]
        if self._comp.shape[0] < 2:
            c = grid[candidates[0]]
            log("Evaluating: " + str(c))
            return candidates[0]

        self._vals = values[complete]
        dimension = self._comp.shape[1]
        durs = durations[complete]

        if not self._is_initialized:
            self._real_init(dimension, self._comp, self._vals, durs)

        #initialize Gaussian processes
        self._initialize_models(durs)

        cand = grid[candidates, :]
        cand = cand[:100]

        mins = self._find_local_minima(self._comp, self._vals, self._models, cand)
        incumbent = self.getIncumbent(self._comp, self._vals, self._models, mins)

        log("Current best: " + str(incumbent))

        #Take candidate that will be optimized
        selected_candidates = cand[:self._num_of_candidates]
        selected_candidates = np.vstack((selected_candidates, mins))

        #Compute entropy of the selected candidates
        overall_entropy = np.zeros(selected_candidates.shape[0])

        for i in xrange(0, len(self._models)):
            overall_entropy += self._entropy_search(selected_candidates,
                                                    self._models[i],
                                                    self._cost_models[i])
        overall_entropy /= len(self._models)

        best_cand = np.argmax(overall_entropy)

        if(self._withPlotting):
            self._visualizer.plot_projected_gp(self._comp, self._vals, self._cost_models[0], True)

            self._visualizer.plot(self._comp, self._vals, self._models[0],
                                                        self._cost_models[0],
                                                        selected_candidates)

        if(self._withPlotting3D):
            self._visualizer.plot3D(self._comp, self._vals, self._models[0],
                                                        self._cost_models[0],
                                                        selected_candidates[best_cand],
                                                        selected_candidates)

        log("Evaluating: " + str(selected_candidates[best_cand]))

        return (len(candidates) + 1, selected_candidates[best_cand])

    def _entropy_search(self, cand, model, cost_model):

        entropy = np.zeros(cand.shape[0])
        if(self._withCosts == True):
            entropy_estimator = EntropyWithCosts(model, cost_model)
            #entropy = map(entropy_estimator.compute, cand)
            for i in xrange(0, cand.shape[0]):
                entropy[i] = entropy_estimator.compute(cand[i])

        else:
            entropy_estimator = Entropy(model)
            entropy = map(entropy_estimator.compute, cand)

        return entropy

    def _initialize_models(self, durs):

        #Slice sampling of hyper parameters
        #Get last sampled hyper-parameters
        (_, noise, amp2, ls) = self._hyper_samples[len(self._hyper_samples) - 1]

        log("last hyper parameters: " +
            str(self._hyper_samples[len(self._hyper_samples) - 1]))

        self._hyper_samples = sample_hyperparameters(self._mcmc_iters,
                                                     self._noiseless,
                                                     self._comp, self._vals,
                                                     self._cov_func, noise,
                                                     amp2, ls)

        (_, noise, amp2, ls) = self._cost_func_hyper_param[len(self._cost_func_hyper_param) - 1]

        self._cost_func_hyper_param = sample_hyperparameters(
                                                self._mcmc_iters,
                                                self._noiseless,
                                                self._comp, durs,
                                                self._cost_cov_func,
                                                noise, amp2, ls)

        self._models = []
        self._cost_models = []

        for h in range(0, len(self._hyper_samples)):
            hyper = self._hyper_samples[h]
            gp = GPModel(self._comp, self._vals, hyper[0], hyper[1],
                         hyper[2], hyper[3], self._covar)

            self._models.append(gp)

            cost_hyper = self._cost_func_hyper_param[h]

            cost_gp = GPModel(self._comp, durs, cost_hyper[0], cost_hyper[1],
                              cost_hyper[2], cost_hyper[3],
                              self._cost_covar)

            self._cost_models.append(cost_gp)

    def getIncumbent(self, evaluated, values, model_list, candidates):
        '''
        Returns the current best guess where the minimum of the objective function could be.
        Args:
            evaluated: numpy matrix containing the points that have been evaluated so far
            values: the corresponding values that have been observed
            model_list: a list of Gaussian processes
            candidates: a numpy matrix of candidates
        Returns:
            a numpy vector that is the point with the highest probability to be the minimum
        '''
        local_minima = self._find_local_minima(evaluated, values, model_list, candidates)
        pmin = self._compute_pmin_probabilities(model_list, candidates)
        return local_minima[np.argmin(pmin)]

    def _compute_pmin_probabilities(self, model_list, candidates):
        '''
        Computes the probability for each candidate to be the minimum. Ideally candidates was computed with
        #_find_local_minima.
        Args:
            model_list: a list of Gaussian process models
            candidates: a numpy matrix of candidates
        Returns:
            a numpy vector containing for each candidate the probability to be the minimum
        '''
        pmin = np.zeros(candidates.shape[0])
        number_of_models = len(model_list)
        for model in model_list:
            m, L = model.getCholeskyForJointSample(candidates)
            #use different Omega for different GPs
            Omega = npr.normal(0, 1, (self._number_of_pmin_samples, candidates.shape[0]))
            pmin += compute_pmin_bins(Omega, m, L) / number_of_models
        return pmin

    def _find_local_minima(self, evaluated, values, model_list, candidates):
        '''
        Tries to find as many local minima as possible of the #_objective_function.
        Args:
            evaluated: numpy matrix containing the points that have been evaluated so far
            values: the corresponding values that have been observed
            model_list: a list of Gaussian processes
            candidates: a numpy matrix of candidates
        Returns:
            a numpy matrix of local minima
        '''
        def objective_function(x, gradients=False):
            if np.any(x < 0) or np.any(x > 1):
                return -np.infty
            return self._objective_function(x, model_list, gradients)
        #TODO: Could it be a good idea start with the maximum instead? To find more local minima?
        #TODO: Actually we should take the best candidate of all candidates in the Sobol sequence.
        starting_point = evaluated[np.argmin(values)]
        # sample points from our proposal measure as starting points for the minimizer
        sampled_points = sample_from_proposal_measure(starting_point, objective_function,
                                     self._incumbent_number_of_minima, self._incumbent_inter_sample_distance)
        #optimization bounds
        opt_bounds = []
        for i in xrange(0, starting_point.shape[0]):
            opt_bounds.append((0, 1))
        minima = []
        for i in range(0, sampled_points.shape[0]):
            optimized = spo.fmin_l_bfgs_b(self._objective_function, sampled_points[i].flatten(), args=(model_list, True),
                                          bounds=opt_bounds, disp=0)
            #we care only for the point, not for the value or debug messages
            optimized = optimized[0]

            #remove duplicates
            append = True
            for j in range(0, len(minima)):
                if np.allclose(minima[j], optimized):
                    append = False
                    break
            if append:
                minima.append(optimized)
        return np.array(minima)

    def _objective_function(self, x, model_list, gradients=False):
        '''
        Computes the sum of mean and standard deviation of each Gaussian process in x.
        Args:
            x: a numpy vector (not a matrix)
            model_list: a list of Gaussian processes
            gradients: whether to compute the gradients
        Returns:
            the value or if gradients is True additionally a numpy vector containing the gradients
        '''
        return self._objective_function_naive(x, model_list, gradients)

    def _objective_function_naive(self, x, model_list, gradients=False):
        '''
        Computes the sum of mean and standard deviation of each Gaussian process in x.
        Args:
            x: a numpy vector (not a matrix)
            model_list: a list of Gaussian processes
            gradients: whether to compute the gradients
        Returns:
            the value or if gradients is True additionally a numpy vector containing the gradients
        '''
        x = np.array([x])
        #will contain mean and standard deviation prediction for each GP
        mean_std = np.zeros([self._mcmc_iters, 2, 1])
        for i in range(0, self._mcmc_iters):
            mean_std[i] = model_list[i].predict(x, True)
        #take square root to get standard deviation
        mean_std[:, 1] = np.sqrt(mean_std[:, 1])
        if not gradients:
            return np.sum(mean_std) / self._mcmc_iters
        mean_std_gradients = np.zeros([self._mcmc_iters, x.shape[1]])
        for i in range(0, self._mcmc_iters):
            mg, vg = model_list[i].getGradients(x[0])
            #getGradient returns the gradient of the variance - to get the gradients of the standard deviation
            # we need to apply chain rule (s(x)=sqrt[v(x)] => s'(x) = 1/2 * v'(x) / sqrt[v(x)]
            stdg = 0.5 * vg / mean_std[i, 1]
            mean_std_gradients[i] = (mg + stdg) / self._mcmc_iters
        #since we want to minimize, we have to turn the sign of the gradient
        return (np.sum(mean_std) / self._mcmc_iters, np.sum(mean_std_gradients, axis=0))
