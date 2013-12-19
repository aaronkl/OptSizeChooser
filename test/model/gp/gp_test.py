'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
from test.abstract_test import AbstractTest, d, scale
import unittest
import numpy as np
import scipy.linalg as spla
import numpy.random as npr
from test.util import cov

class Test(AbstractTest):
    #def setUp(self):
    # in super class
    
    def testPredict(self):
        '''
        Checks if the implementation of the prediction function is correct.
        Assumes that the spear mint implementation is correct.
        '''
        xstar = np.array([scale * npr.randn(d)])
        # The primary covariances for prediction.
        comp_cov   = cov(self, self.X)
        cand_cross = cov(self, self.X, xstar)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(self.X.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        assert(spla.norm(obsv_chol-self.gp._L) == 0)
        alpha = spla.cho_solve((obsv_chol, True), self.y - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        (m,v) = self.gp.predict(xstar, variance=True)
#         print (func_m, m)
#         print (func_v, v)
        assert(abs(func_m-m) < 1e-50)
        assert(abs(func_v-v) < 1e-50)


    def testGetGradients(self):
        '''
        Compares the gradients computed as done originally in spear-mint with our implementation.
        '''
        xstar = np.array([scale * npr.randn(d)])
        cand_cross_grad = self.amp2 * self.cov_grad_func(self.ls, self.X, xstar)
        
        comp_cov   = cov(self, self.X)
        cand_cross = cov(self, self.X, xstar)

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
        assert(spla.norm(mg - grad_xp_m) < 1e-50)
        assert(spla.norm(mv[0] - grad_xp_v[0]) < 1e-50)
        
    def testDrawJointSample(self):
        '''
        Tests how the Gaussian process draws joint samples against a naive implementation.
        '''
        N = npr.randint(1,5)
        Xstar = scale * npr.randn(N,d)
        omega = npr.normal(0,1,N)
        y2 = self.gp.drawJointSample(Xstar, omega)
        
        y1 = np.zeros(N)
        for i in range(0,N):
            y1[i] = self.gp.sample(Xstar[i], omega[i])
            self.gp.update(Xstar[i], y1[i])
        #the naive procedure is numerically unstable
        #that's why we tolerate a higher error here
        assert(spla.norm(y1 - y2) < 1e-5)
        
    def testCopy(self):
        '''
        Asserts that the copy of a GP does indeed not influence the GP it was copied from.
        '''
        xstar = np.array([scale * npr.randn(d)])
        (mu, sigma) = self.gp.predict(xstar, variance=True)
        gp_copy = self.gp.copy()
        x_new = scale * npr.randn(d) #does not need to be a matrix
        y_new = npr.rand()
        gp_copy.update(x_new, y_new)
        (mu2, sigma2) = self.gp.predict(xstar, variance=True)
        assert(np.array_equal(mu, mu2))
        assert(np.array_equal(sigma, sigma2))
        
    
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
