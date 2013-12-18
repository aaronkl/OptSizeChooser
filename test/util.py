'''
Created on Nov 4, 2013

@author: Simon Bartels
'''

import numpy as np
import scipy.linalg as spla
import numpy.random as npr

def makeObservations(dimension, scale):
    #randn samples from N(0,1) so we stretch samples a bit further out for numeric reasons
    N = npr.randint(1,251)
    X = scale * npr.randn(N,dimension)
    y = npr.randn(N)
    return (X,y)

        
def copy_parameters(obj, gp):
    '''
    Copies the parameters of the GP into this object.
    Necessary for the spearmint GP.
    '''
    obj.cov_func = gp._cov_func
    obj.cov_grad_func = gp._covar_derivative
    obj.ls = gp._ls
    obj.noise = gp._noise
    obj.mean = gp._mean
    obj.amp2 = gp._amp2
    
def cov(gp, x1, x2=None):
    '''
    Spearmint covariance function.
    '''
    if x2 is None:
        return gp.amp2 * (gp.cov_func(gp.ls, x1, None)
                           + 1e-6*np.eye(x1.shape[0]))
    else:
        return gp.amp2 * gp.cov_func(gp.ls, x1, x2)