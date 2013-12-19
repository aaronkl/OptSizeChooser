'''
Created on 19.12.2013

@author: bartelss
'''
from test.abstract_test import AbstractTest, scale, d
import unittest
import scipy.optimize as spo
import numpy as np
from acquisition_functions.expected_improvement import ExpectedImprovement
from chooser.OptSizeChooser import _call_minimizer, _compute_ac_gradient_over_hypers, _initialize_acquisition_functions
from support.hyper_parameter_sampling import sample_hyperparameters
from model.gp.gp_model import GPModel


class Test(AbstractTest):

    def setUp(self):
        super(Test, self).setUp()
        _hyper_samples = sample_hyperparameters(20, False, self.X, self.y, 
                                                     self.cov_func, self.noise, self.amp2, self.ls)
        self._models = []
        for h in range(0, len(_hyper_samples)-1):
            hyper = _hyper_samples[h]
            gp = GPModel(self.X, self.y, hyper[0], hyper[1], hyper[2], hyper[3])
            self._models.append(gp)

    def test_call_minimizer(self):
        '''
        Asserts that this function produces indeed something better than the starting point.
        '''
        xstar = np.zeros(d)
        opt_bounds = []# optimization bounds
        for i in xrange(0, d):
            opt_bounds.append((-3*scale, 3*scale))
             
        dimension = (-1, d)
        ac_funcs = _initialize_acquisition_functions(ExpectedImprovement, self.X, self.y, self._models, [])
        x = _call_minimizer(xstar, _compute_ac_gradient_over_hypers, (ac_funcs, dimension), opt_bounds)[0]
        assert(_compute_ac_gradient_over_hypers(x, ac_funcs, dimension)[0] <
               _compute_ac_gradient_over_hypers(xstar, ac_funcs, dimension)[0])
        val1 = 0
        val2 = 0
        for acf in ac_funcs:
            val1+=acf.compute(xstar)
            val2+=acf.compute(x)
        assert(val1<val2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()