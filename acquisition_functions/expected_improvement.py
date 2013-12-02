'''
Created on 18.11.2013

@author: Aaron Klein, Simon Bartels
'''

import numpy as np
import scipy.stats    as sps


class ExpectedImprovement():
    '''
    classdocs
    '''

    def __init__(self, completed_values, func_values):
        '''
        Constructor
        '''
        self._incumbent = np.min(func_values)
        
    def compute(self, candidate, gp, compute_gradient = False):
        
        (func_m, func_v) = gp.predict_vector(candidate)

        # Expected improvement
        func_s = np.sqrt(func_v)
        u = (self._incumbent - func_m) / func_s
        ncdf = sps.norm.cdf(u)
        npdf = sps.norm.pdf(u)
        
        ei = func_s * (u * ncdf + npdf)

        if not compute_gradient:
            return ei
    
        #TODO: This is actually a bit inefficient, since getGradient computes the variance again
        #which we already have here!
        (mg, vg) = gp.getGradients(candidate)
        sg = 0.5 * vg / func_s #we want the gradient of s(x) not of s^2(x)
        
        # this is the result after some simplifications
        grad = npdf * sg + ncdf * mg
        
        #TODO: For which reason ever grad deviates from the spear mint grad by a factor of 2!
        #So, since we assume the spear mint implementation as correct (which might be wrong
        #in this case) we divide by 2 here.
        return (ei, grad[0]/2)
    
    
    def compute_with_prediction(self, func_m, func_v):
        # Expected improvement
        func_s = np.sqrt(func_v)
        u = (self._incumbent - func_m) / func_s
        ncdf = sps.norm.cdf(u)
        npdf = sps.norm.pdf(u)
        
        ei = func_s * (u * ncdf + npdf)

        return ei
                  
        