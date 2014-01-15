'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
from test.abstract_test import AbstractTest, d, scale, cov
import unittest
import numpy as np
import scipy.linalg as spla
import numpy.random as npr
import scipy.stats as sps
from acquisition_functions.expected_improvement import ExpectedImprovement

class Test(AbstractTest):

    def testEI(self):
        Xstar = np.array([scale * npr.randn(d)])
        ei = ExpectedImprovement(self.X, self.y, self.gp, None)
        ei_sp = self._ei_spear_mint(Xstar, self.X, self.y, compute_grad=False)[0]
        ei_val = ei.compute(Xstar[0], False)
        print str(ei_sp) + "=" + str(ei_val)
        assert(abs(ei_val-ei_sp) < 1e-50)
    
    def _ei_spear_mint(self, cand, comp, vals, compute_grad=True):
        """
        Computes EI and gradient of EI as in spear mint.
        """
        best = np.min(vals)

        comp_cov   = cov(self, comp)
        cand_cross = cov(self, self.X, cand)

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
        xstar = scale * npr.random((1,d))
        ei = ExpectedImprovement(self.X, self.y, self.gp, None)
        grad_ei_sp = self._ei_spear_mint(xstar, self.X, self.y, compute_grad=True)[1]
        grad_ei = ei.compute(xstar[0], True)[1]
        
        print str(grad_ei_sp) + "=" + str(grad_ei)
        def f(x):
            return ei.compute(x, False)
        self.assert_first_order_gradient_approximation(f, xstar[0], grad_ei, 1e-13)
        assert(spla.norm(grad_ei-grad_ei_sp) < 1e-50)
            



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testEI']
    unittest.main()
