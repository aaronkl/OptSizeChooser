'''
Created on Oct 28, 2013

@author: Simon Bartels

This class implements the augmented criterion as presented in
"Global Optimization of Stochastic Black-Box Systems via Sequential Kriging Meta-Models"
'''

import numpy as np
import scipy.optimize as spo
import scipy.stats    as sps
import util
from chooser.ConfidenceEIChooser import ConfidenceEIChooser

#TODO: REFACTOR!
def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return BigDataConfidenceEIChooser(expt_dir, **args)

# TODO: write results to file
class BigDataConfidenceEIChooser(ConfidenceEIChooser):
    def __init__(self, parameters):
        '''
        Default constructor.            
        '''
        
        '''The modelfactory.'''
        # TODO: set this via parameter
        self.mf = GPModelFactory
        
        '''The model factory for the cost function.'''
        self.cmf = GPModelFactory
    
    #@override
    def next(self, grid, values, durations,
             candidates, pending, complete):
        comp = grid[complete, :]
        if comp.shape[0] < 2:
            return candidates[0]
        durs = durations[complete]
        self.cost_model = self.cmf.create(comp, durs)
        return super(BigDataConfidenceEIChooser, self).next(grid, values, durations, 
                                                            candidates, pending, complete)
        
    #@override
    def _augmented_ei(self, cand, incumbent, dimension):
        '''
        Overrides the method in the confidence chooser.
        '''
        (aug_ei, grad_aug_ei) = super(BigDataConfidenceEIChooser, self)._augmented_ei(cand, incumbent, dimension)
        cand = np.reshape(cand, dimension)
        cost_augmentation = (1 + self.cost_model.predict(cand))
        (costs_grad, _) = self.cost_model.getGradients(cand)
        aug_ei = aug_ei / cost_augmentation
        #Next step we want to have dEI*(1+costs)^-1 - EI * (1+c(x))^-2 * c'(x)
        #Therefore aug_ei/cost_augmentation correct.
        grad_aug_ei = grad_aug_ei / cost_augmentation - costs_grad * aug_ei / cost_augmentation
        return (aug_ei, grad_aug_ei)