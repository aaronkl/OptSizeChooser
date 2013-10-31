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


d = 2
scale = 25

def _makeObservations():
    #randn samples from N(0,1) so we stretch samples a bit further out for numeric reasons
    N = npr.randint(1,251)
    X = scale * npr.randn(N,d)
    y = npr.randn(N)
    return (X,y)

class Test(unittest.TestCase):


    def setUp(self):
        self.random_state = npr.get_state()
        mf = gp_model_factory.GPModelFactory()
        (X, y) = _makeObservations()
        self.X = X
        self.y = y
        self.gp = mf.create(X, y)
        self._copy_parameters()
        
    def _copy_parameters(self):
        '''
        Copies the parameters of the GP into this object.
        Necessary for the spearmint GP.
        '''
        self.cov = self.gp._covar
        self.cov_grad_func = self.gp._covar_derivative
        self.ls = self.gp._ls
        self.noise = self.gp._noise
        self.mean = self.gp._mean

    def tearDown(self):
        pass


    def testGetGradients(self):
        '''
        Compares the gradients computed as done originally in spear-mint with our implementation.
        '''
        xstar = np.array([scale * npr.randn(d)])
        cand_cross_grad = self.cov_grad_func(self.ls, self.X, xstar)
        
        comp_cov = self.cov(self.ls, self.X)
        cand_cross = self.cov(self.ls, self.X, xstar)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(self.X.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        # Predictive things.
        # Solve the linear systems.
        alpha = spla.cho_solve((obsv_chol, True), self.y - self.mean)

        # Apply covariance function
        grad_cross = np.squeeze(cand_cross_grad)

        grad_xp_m = np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = np.dot(-2 * spla.cho_solve(
                (obsv_chol, True), cand_cross).transpose(), grad_cross)
        
        (mg,mv) = self.gp.getGradients(xstar)
        print (mg,grad_xp_m)
        print (mv, grad_xp_v)
        assert(abs(mg - grad_xp_m) < 0.01)
        assert(abs(mv - grad_xp_v) < 0.01)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()