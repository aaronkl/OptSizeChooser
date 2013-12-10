'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels
'''
import numpy as np
import pickle

class EntropySearch(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        number_of_cand_samples = 10
        #TODO: Use seed
        self._omega = np.random.normal(0, 1, number_of_cand_samples)
    #TODO: actual sampling of the representer points
    def sample_belief_point(self, nb):
        '''
        Parameters in:
            np = number of representer points that we want to sample
        Parameters out:
            set of representer points
        '''
        self._representers = np.zeros(nb)
        self._log_proposal_vals = np.zeros(nb)
        
    def compute(self, candidate, gp, compute_gradient = False):
        pmin = np.zeros(len(self._representers))
        loss = 0
        S = 10
        for o in self._omega:
            y = gp.sample(candidate,o)
            gp_copy = pickle.copy.deepcopy(gp)
            gp_copy.update(candidate, y)
            for s in xrange(0,S):
                vals = gp_copy.draw(self._representers)
                current_min = np.argmin(vals)
                for m in current_min:
                    pmin[m] += 1/(current_min.shape[0])
                
            pmin = pmin / S
            entropy_pmin = - np.dot(pmin, np.log(pmin))
            log_proposal = np.dot(self._log_proposal_vals, pmin)
            
            kl_divergence = entropy_pmin - log_proposal 
            loss = loss + kl_divergence
            
        return loss