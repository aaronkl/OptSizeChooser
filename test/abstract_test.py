'''
Created on 19.12.2013

@author: Simon Bartels

Provides common set up routine for tests.
'''
from __future__ import absolute_import #turns off relative imports
import unittest
import gp
from model.gp.gp_model import GPModel
import support.hyper_parameter_sampling as hps
from test.util import makeObservations, copy_parameters
import numpy.random as npr
import numpy as np

'''
The number of input dimensions for the Gaussian Process.
'''
d = 2

'''
Scale/distribution of the inputs, i.e. factor to the normal distribution.
'''
scale = 25

class AbstractTest(unittest.TestCase):
    def setUp(self):
        self.random_state = npr.get_state()
        #print "random state:"
        #print self.random_state
        (X, y) = makeObservations(d, scale)
        self.X = X
        self.y = y
        covarname = "Matern52"
        cov_func = getattr(gp, covarname)
        noise = 1e-6
        amp2 = 1
        ls = np.ones(d)
        parameter_ls = hps.sample_hyperparameters(15, False, X, y, cov_func, noise, amp2, ls)
        (mean, noise, amp2, ls) = parameter_ls[len(parameter_ls)-1]
        self.gp = GPModel(X, y, mean, noise, amp2, ls, covarname)
        copy_parameters(self, self.gp)

    def tearDown(self):
        pass