'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class extends EntropySearch to be used for big data sets. I.e. it incorporates the cost models
and it ASSUMES that the first value of a point is the data set size.
'''
import numpy as np
import support.pmin_discretization as pmin_discretization
from acquisition_functions.expected_improvement import ExpectedImprovement
from acquisition_functions.entropy_search import EntropySearch, NUMBER_OF_CAND_SAMPLES, NUMBER_OF_REPRESENTER_POINTS

class EntropySearchBigData(EntropySearch):
    def __init__(self, comp, vals, gp, cost_gp=None):
        '''
        Default constructor.
        '''    
        #TODO: Use seed
        self._omega = np.random.normal(0, 1, NUMBER_OF_CAND_SAMPLES)
        self._gp = gp
        self._cost_gp = cost_gp
        starting_point = np.array([comp[np.argmin(vals)][1,]]) #we don't want to slice sample over the first value
        self._ei = ExpectedImprovement(comp, vals, gp, cost_gp)
        self._representers = pmin_discretization.sample_representer_points(starting_point, 
                                                                           self._log_proposal_measure, 
                                                                           NUMBER_OF_REPRESENTER_POINTS)
        
    def compute(self, candidate, compute_gradient = False):
        kl = super(EntropySearchBigData, self).compute(candidate, compute_gradient)
        return kl / self._cost_gp.predict(np.array([candidate]))
    
    def _log_proposal_measure(self, x):
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        #set first value to one     
        x = np.insert(x, 0, 1)
        v = self._ei.compute(x)
        return np.log(v+1e-10)