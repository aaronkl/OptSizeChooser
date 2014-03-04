'''
Created on Oct 29, 2013

@author: Simon Bartels
'''
import unittest
from ..abstract_test import AbstractTest, d, scale, cov
import numpy as np
import numpy.random as npr
from ...acquisition_functions.expected_improvement import ExpectedImprovement
from ...acquisition_functions.entropy_search import EntropySearch, NUMBER_OF_PMIN_SAMPLES, NUMBER_OF_REPRESENTER_POINTS
from ...gp_model import GPModel, fetchKernel, getNumberOfParameters
from ...support import hyper_parameter_sampling as hps

class Test(AbstractTest):

    def testEntropySearch(self):
        Xstar = np.array([scale * npr.randn(d)])
        ei = ExpectedImprovement(self.X, self.y, self.gp, None)
        ei_sp = self._ei_spear_mint(Xstar, self.X, self.y, compute_grad=False)[0]
        ei_val = ei.compute(Xstar[0], False)
        #print str(ei_sp) + "=" + str(ei_val)
        assert(abs(ei_val-ei_sp) < 1e-50)

    def test_pmin_computation(self):
        #f = (x-0.5)^2
        f = lambda(X): np.array([10*(X[i]-0.5)**2 for i in range(0, N)])[:,0]
        d=1

        N = 4

        X = np.zeros([N, d])
        X[0][0] = 0
        X[1][0] = 0.05
        X[2][0] = 0.95
        X[3][0] = 1


        y =  f(X)

        covarname = "Polynomial3"
        cov_func, _ = fetchKernel(covarname)
        noise = 1e-6
        amp2 = 1
        ls = np.ones(getNumberOfParameters(covarname, d))
        parameter_ls = hps.sample_hyperparameters(15, True, X, y, cov_func, noise, amp2, ls)
        (mean, noise, amp2, ls) = parameter_ls[len(parameter_ls)-1]
        gp = GPModel(X, y, mean, noise, amp2, ls, covarname)
        ei = ExpectedImprovement(X, y, gp)
        es = EntropySearch(X, y, gp)

        Omega = np.random.normal(0, 1, (NUMBER_OF_PMIN_SAMPLES,
                                              NUMBER_OF_REPRESENTER_POINTS+1))
        es._representers[0] = np.array([0.5])
        pmin = es._compute_pmin_bins()
        print pmin



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testEI']
    unittest.main()
