'''
Created on Oct 28, 2013

@author: Simon Bartels

This class implements the augmented criterion as presented in
"Global Optimization of Stochastic Black-Box Systems via Sequential Kriging Meta-Models"
'''
#TOD: REFACTOR!
import numpy as np
import scipy.optimize as spo
import scipy.stats    as sps
import util
from model.gp.gp_model_factory import GPModelFactory

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return ConfidenceEIChooser(expt_dir, **args)

#TODO: write results to file
class ConfidenceEIChooser:
    def __init__(self, parameters):
        '''
        Default constructor.
        Args:
            modelfactory: Factory for creating a model, e.g. a Gaussian process (mixture)
        '''
        
        '''The modelfactory.'''
        #TODO: set this via parameter
        self.mf = GPModelFactory()
    
    def next(self, grid, values, durations,
             candidates, pending, complete):
        '''
        Uses only ONE model!
        '''
        comp = grid[complete,:]
        if comp.shape[0] < 2:
            return candidates[0]
        cand = grid[candidates,:]
        vals = values[complete]
        numcand = cand.shape[0]
        dimension = comp.shape[1]
        self.model = self.mf.create(comp, vals)
        #Current best value
        incumbent = self._getIncumbentValue(self.model, comp)
        
        b = []  # optimization bounds
        for i in xrange(0, dimension):
            b.append((0, 1))
        results = [optimize_pt(c, b, incumbent, self) for c in cand]
        best_cand = np.argmax([r[1] for r in results])
        return (int(numcand), results[best_cand][0])
    
    def _getIncumbentValue(self, model, X):
        '''
        Returns the maximal utility value (for the given inputs). 
        '''
        (m,s) = model.predictVariance(X)
        u = -m - s
        i = np.argmin(u)
        return -m[i] - s[i]
    
    def _optimize_pt(self, candidate, b, incumbent):
        '''
        Locally optimizes the augmented EI criterion starting from the candidate.
        Returns a tuple containing the new candidate and it's EI value.
        '''
        dimension = (-1, candidate.shape[0])
        print candidate.flatten()
        
        ret = spo.fmin_l_bfgs_b(self._augmented_ei,
                                candidate.flatten(), args=(incumbent, dimension),
                                bounds=b, disp=0)
        return (ret[0], ret[1])

    
    
    def _augmented_ei(self, cand, incumbent, dimension):
        '''
        Returns value and gradient of the augmented EI criterion.
        Args:
            cand: the candidate
            incumbent: the current best augmented EI value
            dimension: original dimension of the candidate
        '''
        cand = np.reshape(cand, dimension)
        (ei, grad_ei) = ei_calc.expected_improvement(self.model, incumbent, cand, gradient=True)
        (_, v) = self.model.predictVariance(cand)
        #FIXME: fails!
        (_, grad_v) = self.model.getGradients(cand[0])
        noise = self.model.getNoise()
        
        #The augmented EI value
        sqr = np.sqrt(v + noise**2)
        aug = (1 - noise / sqr)
        aug_ei = ei * aug
        
        grad_aug = -noise / 2 * grad_v / sqr**3
        grad_aug_ei = grad_ei * aug + ei * grad_aug #product rule
        return (aug_ei, grad_aug_ei.flatten())