'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class extends EntropySearch to be used for big data sets. I.e. it incorporates the cost models
and it ASSUMES that the first value of a point is the data set size.
'''
import numpy as np
from ..acquisition_functions.expected_improvement import ExpectedImprovement
from ..acquisition_functions.entropy_search import EntropySearch, NUMBER_OF_CANDIDATE_SAMPLES, \
    NUMBER_OF_REPRESENTER_POINTS, NUMBER_OF_PMIN_SAMPLES, sample_representer_points
from scipy.stats import norm
from sobol_lib import i4_sobol_generate


class EntropySearchBigData(EntropySearch):

    def __init__(self, comp, vals, gp, cost_gp=None):

        self._gp = gp
        self._cost_gp = cost_gp
        self._ei = ExpectedImprovement(comp, vals, gp, cost_gp)

        #we don't want to slice sample over the first value
        starting_point = comp[np.argmin(vals)][1:]

        representers = sample_representer_points(starting_point,
                                                 self._sample_measure,
                                                 NUMBER_OF_REPRESENTER_POINTS)

        self._func_sample_locations = np.empty([NUMBER_OF_REPRESENTER_POINTS,
                                                comp.shape[1]])

        #The representer points miss the first coordinate
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            self._func_sample_locations[i] = np.insert(representers[i], 0, 1)

        self._Omega = norm.ppf(i4_sobol_generate(NUMBER_OF_REPRESENTER_POINTS, NUMBER_OF_PMIN_SAMPLES+1, 1)[:,1:]).T
        self._omega_cands = norm.ppf(np.linspace(1. / (NUMBER_OF_CANDIDATE_SAMPLES + 1),
                                           1-1./(NUMBER_OF_CANDIDATE_SAMPLES + 1),
                                           NUMBER_OF_CANDIDATE_SAMPLES))

        self._pmin_old = super(EntropySearchBigData, self)._compute_pmin_bins(gp)

        entropy_pmin_old = -np.dot(self._pmin_old, np.log(self._pmin_old + 1e-10))

        self._log_proposal_vals = np.zeros(NUMBER_OF_REPRESENTER_POINTS)

        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            self._log_proposal_vals[i] = self._log_proposal_measure(self._func_sample_locations[i])

        log_proposal_old = np.dot(self._log_proposal_vals, self._pmin_old)

        self._kl_divergence_old = -(entropy_pmin_old - log_proposal_old) / NUMBER_OF_CANDIDATE_SAMPLES

        self._idx = np.arange(0, NUMBER_OF_PMIN_SAMPLES)

    def compute(self, candidate):

        loss = super(EntropySearchBigData, self).compute(candidate)

        loss = loss - self._kl_divergence_old
        scale = self._cost_gp.predict(np.array([candidate]))

        return loss / scale

    def _sample_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        #set first value to one
        x = np.insert(x, 0, 1)

        return self._log_proposal_measure(x)
