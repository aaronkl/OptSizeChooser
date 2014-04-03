'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

'''
import numpy as np
from spearmint.util import slice_sample
from hyper_parameter_sampling import handle_slice_sampler_exception
from support import compute_expected_improvement
from scipy.stats import norm
from spearmint.sobol_lib import i4_sobol_generate


class Entropy(object):

    def __init__(self, gp):

        #Number of samples for the current candidate
        self._num_of_hallucinated_vals = 300
        #Number of function drawn from the gp
        self._num_of_samples = 300
        #Number of point where pmin is evaluated
        self._num_of_representer_points = 10
        self._gp = gp
        self._idx = np.arange(0, self._num_of_samples)

        #TODO: random sampling
        self._Omega = norm.ppf(i4_sobol_generate(self._num_of_representer_points,
                                    self._num_of_samples + 1, 1)[:, 1:]).T

        self._hallucinated_vals = norm.ppf(np.linspace(1. / (self._num_of_hallucinated_vals + 1),
                                    1 - 1. / (self._num_of_hallucinated_vals + 1),
                                    self._num_of_hallucinated_vals))

        comp = gp.getPoints()
        vals = gp.getValues()
        incumbent = comp[np.argmin(vals)]

        self._sample_representer_points(incumbent)

        self._log_proposal_vals = np.zeros(self._num_of_representer_points)

        for i in range(0, self._num_of_representer_points):
            self._log_proposal_vals[i] = self._log_proposal_measure(self._representer_points[i])

    def _sample_representer_points(self, starting_point):

        self._representer_points = np.zeros([self._num_of_representer_points, starting_point.shape[0]])
        inter_sample_distance = 20 * starting_point.shape[0]
        #TODO: burnin?
        for i in range(0, self._num_of_representer_points):
            #this for loop ensures better mixing
            for c in range(0, inter_sample_distance):
                try:
                    starting_point = slice_sample(starting_point, self._log_proposal_measure)
                except Exception as e:
                    starting_point = handle_slice_sampler_exception(e,
                                                            starting_point,
                                                            self._log_proposal_measure)
            self._representer_points[i] = starting_point

    def _log_proposal_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        v = compute_expected_improvement(x, self._gp)

        return np.log(v + 1e-10)

    def compute(self, candidate):

        kl_divergence = 0

        for i in range(0, self._num_of_hallucinated_vals):

            y = self._gp.sample(candidate, self._hallucinated_vals[i])
            gp_copy = self._gp.copy()
            gp_copy.update(candidate, y)

            pmin = self._compute_pmin(gp_copy)

            entropy_pmin = -np.dot(pmin, np.log(pmin + 1e-50))
            log_proposal = np.dot(self._log_proposal_vals, pmin)

            kl_divergence += (entropy_pmin - log_proposal) / self._num_of_hallucinated_vals

        return -kl_divergence

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
