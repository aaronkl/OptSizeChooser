'''
Created on 19.12.2013

@author: Simon Bartels

Provides common set up routine for tests.
'''
from __future__ import absolute_import #turns off relative imports
import unittest
import gp
from model.gp.gp_model import GPModel, getNumberOfParameters
import support.hyper_parameter_sampling as hps
import numpy.random as npr
import numpy as np
import scipy.linalg as spla

'''
The number of input dimensions for the Gaussian Process.
'''
d = 3

'''
Scale/distribution of the inputs, i.e. factor to the uniform distribution.
'''
scale = 1 #25

def makeObservations(dimension, scale, ground_truth=None):
    #randn samples from N(0,1) so we stretch samples a bit further out for numeric reasons
    N = npr.randint(1,251)
    def mod(x): return x % scale
    vec_mod = np.vectorize(mod)
    #uniformly distributed observations in [0,1) * scale
    X = vec_mod(scale * npr.random((N,dimension))) #% scale
    if not (ground_truth is None):
        y = ground_truth(X)
    else:
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

class AbstractTest(unittest.TestCase):
    def setUp(self):
        seed = npr.randint(65000)
        print("using seed: " + str(seed))
        np.random.seed(seed)
        (X, y) = makeObservations(d, scale)
        self.X = X
        self.y = y
        covarname = "Matern52"
        cov_func = getattr(gp, covarname)
        noise = 1e-6
        amp2 = np.std(y)+1e-4
        ls = np.ones(getNumberOfParameters(covarname, d))
        noiseless = bool(npr.randint(2))
        parameter_ls = hps.sample_hyperparameters(15, noiseless, X, y, cov_func, noise, amp2, ls)
        (mean, noise, amp2, ls) = parameter_ls[len(parameter_ls)-1]
        self.gp = GPModel(X, y, mean, noise, amp2, ls, covarname)
        copy_parameters(self, self.gp)

    def tearDown(self):
        pass
    
    def assert_first_order_gradient_approximation(self, f, x, dfdx, epsilon):
        '''
        Asserts that the computed gradient dfdx has the same sign and is about the same value as the
        first order approxmation.
        Args:
            f: the function
            x: the argument
            dfdx: the first order derivative of f in all arguments of x
            epsilon: the precision
        Returns:
            nothing, makes two assertions
        '''
        first_order_grad_approx = np.zeros(d)
        for i in range(0,d):
            h = np.zeros(d)
            h[i] = epsilon
            first_order_grad_approx[i] = (f(x+h) - f(x-h))/(2*epsilon)
        print "approximation: " + str(first_order_grad_approx) + " computed: " + str(dfdx)
        assert(np.all([np.sign(first_order_grad_approx[i]) == np.sign(dfdx[i]) for i in range(0,d)]))
        assert(spla.norm(first_order_grad_approx - dfdx) < epsilon )
    
    