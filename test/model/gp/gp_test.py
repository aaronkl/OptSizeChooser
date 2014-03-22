'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
import unittest
from ...abstract_test import AbstractTest, d, scale, cov
import numpy as np
import scipy.linalg as spla
import numpy.random as npr

class Test(AbstractTest):
    #def setUp(self):
    # in super class
    
    def testPredict(self):
        '''
        Checks if the implementation of the prediction function is correct.
        Assumes that the spear mint implementation is correct.
        '''
        #TODO: build a test for prediction with matrix as argument!
        #xstar = np.array([scale * npr.randn(d)])
        n = npr.randint(1,5)
        Xstar = npr.random([n,d])
        #xstar = self.X
        # The primary covariances for prediction.
        comp_cov   = cov(self, self.X)
        cand_cross = cov(self, self.X, Xstar)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(self.X.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        assert(spla.norm(obsv_chol-self.gp._L) == 0)
        alpha = spla.cho_solve((obsv_chol, True), self.y - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        (m,v) = self.gp.predict(Xstar, variance=True)
        assert(np.all(v>=0))
#         print (func_m, m)
#         print (func_v, v)
        assert(np.all(abs(func_m-m) < 1e-50))
        print func_v
        print v
        #NOTE: since we calcucalte the variance differently and less efficient we get a higher numerical error
        assert(np.all(abs(func_v-v) < 1e-50))


    def testGetGradients(self):
        '''
        Compares the gradients computed as done originally in spear-mint with our implementation.
        '''
        xstar = scale * npr.random((1,d))
        (mg,vg) = self.gp.getGradients(xstar[0])
        
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
        assert(spla.norm(vg[0] - grad_xp_v[0]) < 1e-50)
        
        #Test against first order approximation
        epsilon = 1e-6
        vg = np.array([vg]) #needs to be in the format [[d0,...,dn]]
        def get_variance(x):
            return self.gp.predict(x, True)[1]
        self.assert_first_order_gradient_approximation(get_variance, xstar, vg, epsilon)
        
        mg = np.array([np.array([mg])]) #we need mg in the format [[d0, d1, ..., dn]]
        def get_mean(x):
            return np.array([self.gp.predict(x)])
        self.assert_first_order_gradient_approximation(self.gp.predict, xstar, mg, epsilon)
        
        
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
        N = npr.randint(1,25)
        Xstar = scale * npr.randn(N,d)
        omega = npr.normal(0,1,N)
        mean, L = self.gp.getCholeskyForJointSample(Xstar)
        y2 = self.gp.drawJointSample(mean, L, omega)
        
        y1 = np.zeros(N)
        for i in range(0,N):
            y1[i] = self.gp.sample(Xstar[i], omega[i])
            self.gp.update(Xstar[i], y1[i])
        #the naive procedure is numerically unstable
        #that's why we tolerate a higher error here
        #TODO: remove
        print self.gp.getNoise()
        print y1
        print y2
        assert(spla.norm(y1 - y2) < 1e-10)
        
        
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
