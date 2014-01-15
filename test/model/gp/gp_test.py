'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
from test.abstract_test import AbstractTest, d, scale, cov
import unittest
import numpy as np
import scipy.linalg as spla
import numpy.random as npr
from model.gp import gp_model

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
        xstar = scale * npr.random((1,d))
        (mg,mv) = self.gp.getGradients(xstar[0])
        
        epsilon = 1e-6
        self.assert_first_order_gradient_approximation(self.gp.predict, xstar, mg, epsilon)
        
        def get_variance(x):
            return self.gp.predict(x, True)[1][0]
        self.assert_first_order_gradient_approximation(get_variance, xstar, mv[0], epsilon)
        
        ######################################################################################
        #Spearmint Code
        #The code below is taken from GPEIOptChooser and adapted to the variables here.
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
        grad_xp_m = np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = -2 * np.dot(spla.cho_solve(
                (obsv_chol, True), cand_cross).transpose(), grad_cross)
        
        ######################################################################################
        #End of Spearmint Code
        
        #it seems the gradient of the spearmint code is already optimized and therefore differs by sign
        #however, the gradient of our implementation agrees with the first order approximation
        grad_xp_m = -grad_xp_m
        grad_xp_v = -grad_xp_v
        assert(spla.norm(mg - grad_xp_m) < 1e-50)
        assert(spla.norm(mv[0] - grad_xp_v[0]) < 1e-50)
        
        
        ######################################################################################
        #This is what Marcus Frean and Philipp Boyle propose in
        # "Using Gaussian Processes to Optimize Expensive Functions."
        
#         #d s(x)/ dx = -(dk/dx)^T K^-1 k / s(x)
#         #=> d v(x)/ dx = d s^2(x)/ dx = 2* s(x) * d s(x)/ dx
#         #=> d v(x)/ dx = -2 * (dk/dx)^T K^-1 k
#         k = cov(self, xstar, self.X)
#         print k.shape
#         print obsv_chol.shape
#         Kk = spla.cho_solve((obsv_chol, True), k.T)
#         dkdx = self.amp2 * self.cov_grad_func(self.ls, xstar, self.X)
#         print dkdx.shape
#         dvdx = -2*np.dot(dkdx[0].T, Kk)
#         print dvdx
        
        
        
    def testDrawJointSample(self):
        '''
        Tests how the Gaussian process draws joint samples against a naive implementation.
        '''
        #TODO: fails too often!
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
        #print y1
        #print y2
        assert(spla.norm(y1 - y2) < 1e-5)
        
    def testCovarianceFunctions(self):
        '''
        Tests the implemented covariance functions.
        '''
        cov_func = gp_model.Polynomial3
        M1 = cov_func(self.ls, self.X, self.X)
        #M1 = M ** 3
        #print M1[3][5]
        #print M[3][5] ** 3
        M1=M1#+1e-6*np.eye(self.X.shape[0])
        M2 = np.zeros([self.X.shape[0], self.X.shape[0]])
        for i in range(0, self.X.shape[0]):
            for j in range(0, self.X.shape[0]):
                M2[i][j] = cov_func(self.ls, self.X[i],self.X[j])
                assert(abs(M1[i][j] -  M2[i][j]) < 1e-5)
        #try:
        spla.cholesky(M1+1e-6*np.eye(self.X.shape[0]), lower=True)
    #    except _:
            
        #spla.cholesky(M2+1e-6*np.eye(M1.shape[0]), lower=True)
        
        
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
