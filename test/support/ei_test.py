'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
import unittest
import numpy as np
import scipy.linalg as spla
import gp
import numpy.random as npr
import scipy.stats as sps
from test.util import *
from model.gp import gp_model_factory
from support.expected_improvement import expected_improvement

d = 2
scale = 25

class Test(unittest.TestCase):


    def setUp(self):
        self.random_state = npr.get_state()
        mf = gp_model_factory.GPModelFactory()
        (X, y) = makeObservations(d, scale)
        self.X = X
        self.y = y
        self.gp = mf.create(X, y)
        copy_parameters(self, self.gp)


    def tearDown(self):
        pass


    def testEI(self):
        incumbent = np.min(self.y)
        Xstar = np.array([scale * npr.randn(d)])
        ei = expected_improvement(self.gp, incumbent, Xstar, gradient=False)[0]
        ei_sp = self._ei_spear_mint(Xstar, self.X, self.y, compute_grad=False)[0]
        assert(abs(ei-ei_sp) < 0.01)
    
    def _ei_spear_mint(self, cand, comp, vals, compute_grad=True):
        """
        Computes EI and gradient of EI as in spear mint.
        """
        best = np.min(vals)

        comp_cov = self.gp._compute_covariance(comp)
        cand_cross = self.amp2 * self.cov(self.ls, comp, cand)

        # Compute the required Cholesky.
        obsv_cov = comp_cov + self.noise * np.eye(comp.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)

        # Predictive things.
        # Solve the linear systems.
        alpha = spla.cho_solve((obsv_chol, True), vals - self.mean)
        beta = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)

        # Expected improvement
        func_s = np.sqrt(func_v)
        u = (best - func_m) / func_s
        ncdf = sps.norm.cdf(u)
        npdf = sps.norm.pdf(u)
        ei = func_s * (u * ncdf + npdf)

        if not compute_grad:
            return ei
        
        cand_cross_grad = self.amp2 * self.cov_grad_func(self.ls, comp, cand)

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -ncdf
        g_ei_s2 = 0.5 * npdf / func_s

        # Apply covariance function
        grad_cross = np.squeeze(cand_cross_grad)

        grad_xp_m = np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = np.dot(-2 * spla.cho_solve(
                (obsv_chol, True), cand_cross).transpose(), grad_cross)

        grad_xp = 0.5 * self.amp2 * (grad_xp_m * g_ei_m + grad_xp_v * g_ei_s2)
        return ei, grad_xp[0]
    
    def testGradientEI(self):
        incumbent = np.min(self.y)
        xstar = np.array([scale * npr.randn(d)])
        ei = expected_improvement(self.gp, incumbent, xstar, gradient=True)[1]
        ei_sp = self._ei_spear_mint(xstar, self.X, self.y, compute_grad=True)[1]
        print (ei, ei_sp)
        assert(spla.norm(ei-ei_sp) < 1e-50)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testEI']
    unittest.main()
