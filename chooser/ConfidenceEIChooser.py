'''
Created on Oct 28, 2013

@author: Simon Bartels

This class implements the augmented criterion as presented in
"Global Optimization of Stochastic Black-Box Systems via Sequential Kriging Meta-Models"
'''

import numpy as np
import scipy.stats    as sps
import scipy.optimize as spo
import util
import support.expected_improvement as ei_calc


def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return ConfidenceEIChooser(expt_dir, **args)

'''
Returns the maximal utility value (for the given inputs). 
'''
def getIncumbentValue(model, X):
    (m,s) = model.predictVariance(X)
    u = -m - s
    i = np.argmin(u)
    return -m[i] - s[i]

def optimize_pt(candidate, b, model, incumbent):
    '''
    Locally optimizes the augmented EI criterion starting from the candidate.
    '''
    def wrapper(cand):
        cand = np.reshape(cand, (-1, candidate.shape[0]))
        return ei_calc.ExpectedImprovement(model, incumbent, cand, gradient=True)
    ret = spo.fmin_l_bfgs_b(wrapper,
                            candidate.flatten(),
                            bounds=b, disp=0)
    return (ret[0], ret[1])

class ConfidenceEIChooser:
    def __init__(self, modelfactory):
        '''The modelfactory.'''
        self.mf = modelfactory
    
    # Given a set of completed 'experiments' in the unit hypercube with
    # corresponding objective 'values', pick from the next experiment to
    # run according to the acquisition function.
    def next(self, grid, values, durations,
             candidates, pending, complete):
        '''
        Uses only ONE model!
        '''
        comp = grid[complete,:]
        cand = grid[candidates,:]
        vals = values[complete]
        numcand = cand.shape[0]
        dimension = comp.shape[1]
        model = self.mf.create(comp, vals)
        model.optimize()
        #Current best value
        incumbent = getIncumbentValue(model, comp)
        
        b = []  # optimization bounds
        for i in xrange(0, dimension):
            b.append((0, 1))
        results = [optimize_pt(c, b, model, incumbent) for c in cand]
        best_cand = np.argmax([r[1] for r in results])
        return (int(numcand), results[best_cand][0])