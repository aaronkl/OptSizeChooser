'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class extends EntropySearch to be used for big data sets. I.e. it incorporates the cost models
and it ASSUMES that the first value of a point is the data set size.
'''
import numpy as np
#import support.pmin_discretization as pmin_discretization
from ..acquisition_functions.expected_improvement import ExpectedImprovement
from ..acquisition_functions.entropy_search import EntropySearch, NUMBER_OF_CAND_SAMPLES, NUMBER_OF_REPRESENTER_POINTS

class EntropySearchBigData(EntropySearch):
    def __init__(self, comp, vals, gp, cost_gp=None):
        '''
        Default constructor.
        '''
        self._omega = np.random.normal(0, 1, NUMBER_OF_CAND_SAMPLES)
        self._gp = gp
        self._cost_gp = cost_gp
        starting_point = np.array([comp[np.argmin(vals)][1,]]) #we don't want to slice sample over the first value
        self._ei = ExpectedImprovement(comp, vals, gp, cost_gp)
        representers = self.sample_representer_points(starting_point, 
                                                                           self._sample_measure, 
                                                                           NUMBER_OF_REPRESENTER_POINTS)
        self._representers = np.empty([NUMBER_OF_REPRESENTER_POINTS, comp.shape[1]])
        #the representers miss the first coordinate: we need to add it here.
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            self._representers[i] = np.insert(representers[i], 0, 1) #set first value to one
        
    def compute(self, candidate, compute_gradient = False):
        kl = super(EntropySearchBigData, self).compute(candidate, compute_gradient)
        #TODO: How to scale the Entropy reduction?
        #It is basically a question of how long one is willing to wait for how much improvement
        #return kl / np.log(self._cost_gp.predict(np.array([candidate]))+1)
        #TODO: change back
        scale = np.exp(self._cost_gp.predict(np.array([candidate])))
        if scale == 0:
            return np.inf
        return kl / scale
    
    def _sample_measure(self, x):
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        #set first value to one     
        x = np.insert(x, 0, 1)
        return self._log_proposal_measure(x)
