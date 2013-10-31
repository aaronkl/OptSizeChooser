'''
Created on Oct 29, 2013

@author: raven
'''
import unittest


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def testEI(self):
        pass
    
    def testGradientEI(self):
        cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
        cand_cross_grad = cov_grad_func(self.ls, comp, cand)

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -ncdf
        g_ei_s2 = 0.5 * npdf / func_s

        # Apply covariance function
        grad_cross = np.squeeze(cand_cross_grad)

        grad_xp_m = np.dot(alpha.transpose(), grad_cross)
        grad_xp_v = np.dot(-2 * spla.cho_solve(
                (obsv_chol, True), cand_cross).transpose(), grad_cross)

        grad_xp = 0.5 * self.amp2 * (grad_xp_m * g_ei_m + grad_xp_v * g_ei_s2)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testEI']
    unittest.main()