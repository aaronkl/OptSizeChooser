'''
Created on 12.12.2013

@author: Simon Bartels

Tests the unit support.pmin_discretization.
'''
import unittest
import support.pmin_discretization as pmin
import acquisition_functions.expected_improvement as EI
from test.util import *
import numpy.random as npr
import model.gp.gp_model_factory as gp_model_factory

'''
Dimension of the points.
'''
d = 2


'''
Scale of the input points, i.e. each point is likely to be in the range [-scale,scale]
'''
scale = 25


class Test(unittest.TestCase):


    def setUp(self):
        self.random_state = npr.get_state()
        mf = gp_model_factory.GPModelFactory()
        (X, y) = makeObservations(d, scale)
        self.X = X
        self.y = y
        self.gp = mf.create(X, y)

    def tearDown(self):
        pass


    def testRepresenterPoints(self):
        starting_point = scale * npr.randn(1,d)[0] #randn returns a matrix
        ei = EI.ExpectedImprovement(self.y)
        print("starting with:" + str(starting_point))
        print("log EI: " + str(ei.compute(starting_point, self.gp)))
        #TODO: - or not -? log or not?
        def log_proposal_measure(x):
            if np.any(x < -3*scale) or np.any(x > 3*scale):
                    return -np.inf            
            v = ei.compute(x, self.gp)
            return np.log(v)
        number_of_representer_points = npr.randint(1,25)
        (r, v) = pmin.sample_representer_points(starting_point, log_proposal_measure, number_of_representer_points)
        print(r)
        print(v)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()