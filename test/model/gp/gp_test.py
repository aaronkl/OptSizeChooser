'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
import unittest
import numpy as np
import scipy.linalg as spla
import gp
import numpy.random as npr
from model.gp import gp_model_factory

def _makeObservations():
    d = 4
    N = npr.randint(1,251)
    

class Test(unittest.TestCase):


    def setUp(self):
        self.random_state = npr.get_state()
        mf = gp_model_factory.GPModelFactory()
        (X, y) = _makeObservations()
        self.gp = mf.create(X, y)


    def tearDown(self):
        pass


    def testGetGradients(self):
        cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
        cand_cross_grad = cov_grad_func(self.ls, comp, cand)

        # Apply covariance function
        grad_cross = np.squeeze(cand_cross_grad)

        grad_xp_m = np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = np.dot(-2 * spla.cho_solve(
                (obsv_chol, True), cand_cross).transpose(), grad_cross)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()