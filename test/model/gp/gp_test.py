'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
import unittest
import numpy as np
import scipy.linalg as spla
import gp
import numpy.random as npr
from test.util import *
from model.gp import gp_model_factory

'''
The number of input dimensions for the Gaussian Process.
'''
d = 2

'''
Scale/distribution of the inputs, i.e. factor to the normal distribution.
'''
scale = 25

class Test(unittest.TestCase):


    def setUp(self):
        self.random_state = npr.get_state()
        #print "random state:"
        #print self.random_state
        mf = gp_model_factory.GPModelFactory()
        (X, y) = makeObservations(d, scale)
        self.X = X
        self.y = y
        self.gp = mf.create(X, y)
        copy_parameters(self, self.gp)

    def tearDown(self):
        pass
    
    def testPredict(self):
        '''
        Checks if the implementation of the prediction function is correct.
        Assumes that the spear mint implementation is correct.
        '''
        xstar = np.array([scale * npr.randn(d)])
                # The primary covariances for prediction.
        comp_cov = self.amp2 * self.cov(self.ls, self.X)
        cand_cross = self.amp2 * self.cov(self.ls, self.X, xstar)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(self.X.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        alpha = spla.cho_solve((obsv_chol, True), self.y - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        (m,v) = self.gp.predict(xstar, variance=True)
#         print (func_m, m)
#         print (func_v, v)
        assert(abs(func_m-m) < 0.0001)
        assert(abs(func_v-v) < 0.0001)


    def testGetGradients(self):
        '''
        Compares the gradients computed as done originally in spear-mint with our implementation.
        '''
        xstar = np.array([scale * npr.randn(d)])
        cand_cross_grad = self.amp2 * self.cov_grad_func(self.ls, self.X, xstar)
        
        comp_cov = self.gp._compute_covariance(self.X)
        cand_cross = self.amp2 * self.cov(self.ls, self.X, xstar)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(self.X.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        # Predictive things.
        # Solve the linear systems.
        alpha = spla.cho_solve((obsv_chol, True), self.y - self.mean)

        # Apply covariance function
        grad_cross = np.squeeze(cand_cross_grad)
        grad_xp_m = -np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = -2 * np.dot(spla.cho_solve(
                (obsv_chol, True), cand_cross).transpose(), grad_cross)
               
        (mg,mv) = self.gp.getGradients(xstar[0])
        print (mg, grad_xp_m)
        print (mv, grad_xp_v)
        assert(spla.norm(mg - grad_xp_m) < 1e-50)
        assert(spla.norm(mv[0] - grad_xp_v[0]) < 1e-50)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
