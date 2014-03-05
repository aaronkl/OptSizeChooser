'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class extends EntropySearch to be used for big data sets. I.e. it incorporates the cost models
and it ASSUMES that the first value of a point is the data set size.
'''
import numpy as np
from ..acquisition_functions.expected_improvement import ExpectedImprovement
from ..acquisition_functions.entropy_search import EntropySearch, NUMBER_OF_CAND_SAMPLES, \
    NUMBER_OF_REPRESENTER_POINTS, NUMBER_OF_PMIN_SAMPLES, sample_representer_points, \
    NUMBER_OF_SAMPLE_LOCATIONS


class EntropySearchBigData(EntropySearch):
    def __init__(self, comp, vals, gp, cost_gp=None):
        '''
        Default constructor.

        starting_point = comp[np.argmin(vals)]
        self._func_sample_locations = np.empty([NUMBER_OF_SAMPLE_LOCATIONS,starting_point.shape[0]])
        self._func_sample_locations[NUMBER_OF_SAMPLE_LOCATIONS-NUMBER_OF_REPRESENTER_POINTS:] \
            = sample_representer_points(starting_point,self._log_proposal_measure,
                                        NUMBER_OF_REPRESENTER_POINTS)


        '''
        self._Omega = np.random.normal(0, 1, (NUMBER_OF_PMIN_SAMPLES,
                                              NUMBER_OF_SAMPLE_LOCATIONS))
        self._gp = gp
        self._cost_gp = cost_gp
        self._ei = ExpectedImprovement(comp, vals, gp, cost_gp)

        #we don't want to slice sample over the first value
        starting_point = comp[np.argmin(vals)][1:]

        representers = sample_representer_points(starting_point,
                                                 self._sample_measure,
                                                 NUMBER_OF_REPRESENTER_POINTS)

        #We add an additional the candidate to the representer points
        #where the functions will be evaluated
        self._func_sample_locations = np.empty([NUMBER_OF_SAMPLE_LOCATIONS, comp.shape[1]])
        #The representer points miss the first coordinate: we need to add it here.
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            self._func_sample_locations[i + 1] = np.insert(representers[i], 0, 1) #set first value to one

    def compute(self, candidate, compute_gradient=False):
        loss = super(EntropySearchBigData, self).compute(candidate, compute_gradient)

        scale = self._cost_gp.predict(np.array([candidate]))
#         print "scale: " + str(scale)
#         print "kl: " + str(kl)
        if scale == 0:
            return np.inf
        #return kl / scale
        return loss * scale

    def _sample_measure(self, x):
        if np.any(x < 0) or np.any(x > 1):
            return -np.inf
        #set first value to one
        x = np.insert(x, 0, 1)
        return self._log_proposal_measure(x)
