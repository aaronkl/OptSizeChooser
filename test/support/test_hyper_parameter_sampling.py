'''
Created on 12.12.2013

@author: Simon Bartels

Tests the unit support.hyper_parameter_sampling.
'''
from __future__ import absolute_import #turns off relative imports
from chooser.GPEIOptChooser import GPEIOptChooser
from test.abstract_test import AbstractTest, d
import unittest
import numpy as np
import numpy.random as npr
from support.hyper_parameter_sampling import sample_hyperparameters
import gp
from sre_compile import _ASSERT_CODES

class Test(AbstractTest):


    def testSpearmintEquivalence(self):
        '''
        Tests that our implementation generates the same samples as the spearmint version.
        '''
        mcmc_iters = 10
        noiseless = False
        ls = np.ones(d)
        amp2 = np.std(self.y)+1e-4
        noise = 1e-3
        mean = np.mean(self.y)
        covar_name = "Matern52"
        
        
        seed = npr.randint(65000)
        np.random.seed(seed)
        expt_dir = "."
        sp_sampling = GPEIOptChooser(expt_dir,covar_name, mcmc_iters)
        sp_sampling.noise = noise
        sp_sampling.mean = mean
        sp_sampling.amp2 = amp2
        sp_sampling.ls = ls
        sp_sampling.D = d
        for mcmc_iter in xrange(mcmc_iters):
            sp_sampling.sample_hypers(self.X, self.y)
        sp_samples = sp_sampling.hyper_samples
        
        #reset random number generation
        np.random.seed(seed)
        samples = sample_hyperparameters(mcmc_iters, noiseless, self.X, self.y, getattr(gp, covar_name), noise, amp2, ls)
        self._assert_all_samples_equal(sp_samples, samples)
        
    def _assert_all_samples_equal(self, ls1, ls2):
        #assert that all samples equal
        def samples_equals(s1, s2):
            val = True
            for i in range(0, 2):
                val = val and s1[i] == s2[i]
            for i in range(0, d):
                val = val and s1[3][i] == s2[3][i]
            assert(val)
            return val
        
        for i in range(0, max(len(ls1), len(ls2))):
            assert(samples_equals(ls1[i], ls2[i]))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()