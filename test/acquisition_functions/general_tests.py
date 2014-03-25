'''
Created on Oct 31, 2013

@author: Simon Bartels

This class contains tests generally applying to all acquisition functions. I.e. the tests in here
model expected behaviour from the perspective of the chooser.
'''
import unittest


class Test(unittest.TestCase):


    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_is_state_free(self):
        '''
        Calls the acquisition function twice with the same argument. This MUST yield the same value.
        '''
        #TODO: Implement
        raise NotImplementedError("TODO: Implement")
    
    def test_reentrant(self):
        '''
        Tests that the acquisition functions can be used in parallel.
        '''
        #TODO: implement

    def test_does_not_touch_gp(self):
        '''
        Asserts that acquisition functions don't change the given Gaussian process. Necessary since they share it.
        '''


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()