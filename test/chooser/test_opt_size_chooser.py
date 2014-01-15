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
import scipy.linalg as spla
#from Visualizer import plot_one_dimensional_gp
import matplotlib.pyplot as plt
plt.ion()
plt.show()

class Test(AbstractTest):

    def setUp(self):
        super(Test, self).setUp()
        _hyper_samples = sample_hyperparameters(20, True, self.X, self.y, 
                                                     self.cov_func, self.noise, self.amp2, self.ls)#[100-1]]
        print _hyper_samples
        self._models = []
        for h in range(0, len(_hyper_samples)):
            hyper = _hyper_samples[h]
            gp = GPModel(self.X, self.y, hyper[0], hyper[1], hyper[2], hyper[3])
            plot_one_dimensional_gp(self.X, self.y, gp)
            self._models.append(gp)

    def test_call_minimizer(self):
        '''
        Asserts that this function produces indeed something better than the starting point.
        '''
        xstar = float(scale)/2 * np.ones(d)
        opt_bounds = []# optimization bounds
        for i in xrange(0, d):
            opt_bounds.append((0, scale))
             
        dimension = (-1, d)
        ac_funcs = _initialize_acquisition_functions(ExpectedImprovement, self.X, self.y, self._models, [])
        x = _call_minimizer(xstar, _compute_ac_gradient_over_hypers, (ac_funcs, dimension), opt_bounds)[0]
        print x
        print xstar
        xrand = xstar + np.ones(d)*0.1
        print _compute_ac_gradient_over_hypers(xrand, ac_funcs, dimension)
        print _compute_ac_gradient_over_hypers(xstar, ac_funcs, dimension)
        print _compute_ac_gradient_over_hypers(x, ac_funcs, dimension)
        val1 = 0
        val2 = 0
        for acf in ac_funcs:
            val1+=acf.compute(xstar)
            val2+=acf.compute(x)
        print val1
        print val2
        assert(val1<val2)
        assert(spla.norm(x-xstar)>0)
        assert(_compute_ac_gradient_over_hypers(x, ac_funcs, dimension)[0] <
               _compute_ac_gradient_over_hypers(xstar, ac_funcs, dimension)[0])

        
    #def test_local_optimization(self):
        #TODO: implement
    #    raise NotImplementedError("to be implemented")


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    
def plot_one_dimensional_gp(X, y, model, acquisition_function=None, acquisition_points=None):
    plt.figure(2)
    plt.plot(X[:], y, 'g+')

    x = np.linspace(0, 1, 100)[:, np.newaxis]
    # all inputs are normalized by spearmint automatically, thus max_size = 1
    test_inputs = np.ones((100, 1))
    for i in range(0, 100):
        test_inputs[i] = x[i]

    mean = model.predict(test_inputs, False)
    
    plt.plot(x, mean, 'b')
    plt.grid(True)
    
    if acquisition_function is not None:
        acquisition_values = np.zeros(acquisition_points.shape[0])
        for i in range(0, acquisition_points.shape[0]):
            acquisition_values[i] = acquisition_function(acquisition_points[i])
        plt.plot(acquisition_points, acquisition_values, 'o')

    plt.draw()
    plt.ion()
    plt.show()