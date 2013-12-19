'''
Created on 12.12.2013

@author: Simon Bartels

Tests the unit support.pmin_discretization.
'''
from test.abstract_test import AbstractTest, d, scale
import unittest
import support.pmin_discretization as pmin
import acquisition_functions.expected_improvement as EI
import numpy as np
import numpy.random as npr

class Test(AbstractTest):


    def testRepresenterPoints(self):
        starting_point = scale * npr.randn(1,d)[0] #randn returns a matrix
        ei = EI.ExpectedImprovement()
        ei.initialize(self.X, self.y, self.gp, None)
        print("starting with:" + str(starting_point))
        print("log EI: " + str(ei.compute(starting_point, self.gp)))
        #TODO: - or not -? log or not?
        def log_proposal_measure(x):
            if np.any(x < -3*scale) or np.any(x > 3*scale):
                    return -np.inf            
            v = ei.compute(x)
            return np.log(v)
        number_of_representer_points = npr.randint(1,25)
        r = pmin.sample_representer_points(starting_point, log_proposal_measure, number_of_representer_points)
        print r

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()