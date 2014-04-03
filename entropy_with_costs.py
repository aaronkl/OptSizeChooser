'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class extends EntropySearch to be used for big data sets. I.e. it incorporates the cost models
and it ASSUMES that the first value of a point is the data set size.
'''
import numpy as np
from support import compute_expected_improvement
from scipy.stats import norm
from spearmint.sobol_lib import i4_sobol_generate
from spearmint.util import slice_sample
from hyper_parameter_sampling import handle_slice_sampler_exception


class EntropyWithCosts():

    def __init__(self, gp, cost_gp):

        self._gp = gp
        self._cost_gp = cost_gp
        self._num_of_hallucinated_vals = 300
        self._num_of_samples = 300
        self._num_of_representer_points = 10

        comp = gp.getPoints()
        vals = gp.getValues()

        starting_point = comp[np.argmin(vals)][1:]
        self._sample_representer_points(starting_point)

        self._Omega = norm.ppf(i4_sobol_generate(self._num_of_representer_points, self._num_of_samples + 1, 1)[:, 1:]).T

        self._omega_cands = norm.ppf(np.linspace(1. / (self._num_of_hallucinated_vals + 1),
                                           1 - 1. / (self._num_of_hallucinated_vals + 1),
                                           self._num_of_hallucinated_vals))

        #TODO: Change it to vector computation
        self._pmin_old = self._compute_pmin(self._gp)
        entropy_pmin_old = -np.dot(self._pmin_old, np.log(self._pmin_old + 1e-50))

        self._log_proposal_vals = np.zeros(self._num_of_representer_points)
        for i in range(0, self._num_of_representer_points):
            self._log_proposal_vals[i] = self._log_proposal_measure(self._representer_points[i])

        log_proposal_old = np.dot(self._log_proposal_vals, self._pmin_old)
        self._kl_divergence_old = -(entropy_pmin_old - log_proposal_old)

        self._idx = np.arange(0, self._num_of_samples)

    def _sample_representer_points(self, starting_point):

        points = np.zeros([self._num_of_representer_points, starting_point.shape[0]])
        inter_sample_distance = 20 * starting_point.shape[0]

        for i in range(0, self._num_of_representer_points):

            for c in range(0, inter_sample_distance):
                try:
                    starting_point = slice_sample(starting_point,
                                                  self._sample_measure)
                except Exception as e:
                    starting_point = handle_slice_sampler_exception(e,
                                                            starting_point,
                                                            self._sample_measure)
            points[i] = starting_point

        self._representer_points = np.empty([self._num_of_representer_points, starting_point.shape[0] + 1])

        for i in range(0, self._num_of_representer_points):
            self._representer_points[i] = np.insert(points[i], 0, 1)

    def compute(self, candidate):

        kl_divergence = 0

        for i in range(0, self._num_of_hallucinated_vals):

            y = self._gp.sample(candidate, self._omega_cands[i])
            gp_copy = self._gp.copy()
            gp_copy.update(candidate, y)

            pmin = self._compute_pmin(gp_copy)

            entropy_pmin = -np.dot(pmin, np.log(pmin + 1e-50))
            log_proposal = np.dot(self._log_proposal_vals, pmin)

            kl_divergence += (entropy_pmin - log_proposal) / self._num_of_hallucinated_vals

        kl_divergence = -kl_divergence - self._kl_divergence_old

        scale = self._cost_gp.predict(np.array([candidate]))
        scale = np.max([1e-50, scale])

        return kl_divergence / scale

    def _sample_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        #set first value to one
        x = np.insert(x, 0, 1)

        v = compute_expected_improvement(x, self._gp)

        return np.log(v + 1e-10)

    def _log_proposal_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        v = compute_expected_improvement(x, self._gp)

        return np.log(v + 1e-10)

#TODO: Take it out of the class and put it in support

    def _compute_pmin(self, gp):

        pmin = np.zeros(self._num_of_representer_points)
        mean, L = gp.getCholeskyForJointSample(self._representer_points)

        for omega in self._Omega:

            vals = gp.drawJointSample(mean, L, omega)
            mins = np.where(vals == vals.min())[0]
            number_of_mins = len(mins)

            for m in mins:
                pmin[m] += 1. / (number_of_mins)

        pmin = pmin / self._num_of_hallucinated_vals

        return pmin
