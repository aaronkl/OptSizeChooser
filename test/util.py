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
    obj.cov = gp._covar
    obj.cov_grad_func = gp._covar_derivative
    obj.ls = gp._ls
    obj.noise = gp._noise
    obj.mean = gp._mean
    obj.amp2 = gp._amp2