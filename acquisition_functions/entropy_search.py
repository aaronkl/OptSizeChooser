'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class implements an acquisition function similar to what is proposed in 
"Entropy Search for Information-Efficient Global Optimization" by Hennig and Schuler in 2013.
Instead of using Expectation Propagation we apply sampling.
'''
import numpy as np
import copy
import support.pmin_discretization as pmin_discretization

'''
The number of points used to represent/discretize Pmin.
'''
NUMBER_OF_REPRESENTER_POINTS = 20

'''
The number of independent samples drawn for one candidate to approximate Pmin.  
'''
NUMBER_OF_CAND_SAMPLES = 10

'''
The number of independent joint samples drawn for the representer points.
'''
NUMBER_OF_PMIN_SAMPLES = 10

class EntropySearch():
    def __init__(self, log_proposal_measure, starting_point):
        '''
        Default constructor.
        Args:
            log_proposal_measure: A function that measures in log-scale how suitable a point is to represent Pmin. 
        '''
        #TODO: Use seed
        self._omega = np.random.normal(0, 1, NUMBER_OF_CAND_SAMPLES)
        (r, lv) = pmin_discretization.sample_representer_points(starting_point, log_proposal_measure, NUMBER_OF_REPRESENTER_POINTS)
        self._representers = r
        self._log_proposal_vals = lv
    def compute(self, candidate, gp, compute_gradient = False):
        if compute_gradient:
            raise NotImplementedError("computing gradients not supported by this acquisition function")
        pmin = np.zeros(len(self._representers))
        loss = 0
        
        for o in self._omega:
            y = gp.sample(candidate,o)
            gp_copy = copy.deepcopy(gp)
            gp_copy.update(candidate, y)
            for s in xrange(0,NUMBER_OF_PMIN_SAMPLES):
                vals = gp_copy.draw(self._representers)
                #FIXME: Seems like we are using an old version of numpy => argmin returns a number!
                current_min = np.argmin(vals)
                #for m in current_min:
                #    pmin[m] += 1/(current_min.shape[0])
                
                #This is the fix:
                pmin[current_min]+=1
            pmin = pmin / NUMBER_OF_PMIN_SAMPLES
            entropy_pmin = - np.dot(pmin, np.log(pmin))
            log_proposal = np.dot(self._log_proposal_vals, pmin)
            
            kl_divergence = entropy_pmin - log_proposal 
            loss = loss + kl_divergence
            
        return loss