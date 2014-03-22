'''
Created on Oct 29, 2013

@author: Simon Bartels

Test to assert certain properties of the Entropy Search acquisition function.
'''
from mercurial.store import encodedstore
import unittest
from ..abstract_test import AbstractTest, d, scale, cov
import numpy as np
import numpy.random as npr
from ...acquisition_functions.expected_improvement import ExpectedImprovement
from ...acquisition_functions.entropy_search import EntropySearch, NUMBER_OF_PMIN_SAMPLES, \
    NUMBER_OF_REPRESENTER_POINTS, NUMBER_OF_CANDIDATE_SAMPLES
from ...acquisition_functions import entropy_search
from ...gp_model import GPModel, fetchKernel, getNumberOfParameters
from ...support import hyper_parameter_sampling as hps
import time

def _compute_entropy(es, candidate):
    '''
    Computes the entropy of Pmin aftger evaluting in a candidate using the compute function of EntropySearch.
    Args:
        es: an initialized entropy search acquisition function
        candidate: the candidate (numpy vector)
    Returns:
        the expected entropy in Pmin after an evaluation of the objective function in the candidate
    '''
    kl = -es.compute_naive(candidate) #we need to turn the sign
    #to get the entropy we have to denormalize and add the pmin^T*log(vals) terms
    entropy = kl * entropy_search.NUMBER_OF_CANDIDATE_SAMPLES
    for i in range(0, entropy_search.NUMBER_OF_CANDIDATE_SAMPLES):
        gp2 = es._gp.copy()
        gp2.update(candidate, gp2.sample(candidate, es._omega_cands[i]))
        pmin = es._compute_pmin_bins(gp2)
        entropy += np.dot(pmin, es._log_proposal_vals)
    #now we need to normalize again
    entropy = entropy / entropy_search.NUMBER_OF_CANDIDATE_SAMPLES
    return entropy

def _create_setup():
    '''
    Sets the observed datapoints and intiializes Gaussian process and Entropy Search acquisition function.
    Returns:
        (X, y, gp, es): i.e. the datapoints, the observations, the Gaussian process and Entropy Search function
    '''
    #we have two datapoints in 0.2 and 0.8 both with value 0
    dim=1
    N = 2
    X = np.zeros([N, dim])
    y = np.zeros(N)
    X[0][0] = 0.2
    y[0] = 0
    X[1][0] = 0.8
    y[1] = 0
    covarname = "ARDSE"
    cov_func, _ = fetchKernel(covarname)
    mean = 0
    noise = 1e-6
    amp2 = 1
    ls = np.ones(getNumberOfParameters(covarname, dim))/4
    gp = GPModel(X, y, mean, noise, amp2, ls, covarname)
    #entropy_search.NUMBER_OF_PMIN_SAMPLES = 200
    entropy_search.NUMBER_OF_REPRESENTER_POINTS = 2
    es = EntropySearch(X, y, gp)
    #Now we initialize EntropySearch as we need it
    # one representer in 0.2 (i.e. an observed point)
    # and one in 0.5 (where the GP has the most variance)
    es._func_sample_locations[0] = X[0]
    es._func_sample_locations[1][0] = (X[0]+X[1])/2
    #and we need to update the values of the representer points
    for i in range(0, 2):
        es._log_proposal_vals[i] = es._log_proposal_measure(es._func_sample_locations[i])
    return (X, y, gp, es)

#TODO: make sure entropy search also works in the noisy case!

class Test(AbstractTest):
    def setUp(self):
        #don't know if changes to global variables are permanent across tests but I guess it can't hurt to reset
        entropy_search.NUMBER_OF_CANDIDATE_SAMPLES = NUMBER_OF_CANDIDATE_SAMPLES
        entropy_search.NUMBER_OF_REPRESENTER_POINTS = NUMBER_OF_REPRESENTER_POINTS
        entropy_search.NUMBER_OF_PMIN_SAMPLES = NUMBER_OF_PMIN_SAMPLES
        #call super setup
        super(Test, self).setUp()

    def test_faster_implementation(self):
        #entropy_search.NUMBER_OF_CANDIDATE_SAMPLES = 3
        es = EntropySearch(self.X, self.y, self.gp)
        #es._omega_cands[0] = 2
        cand = np.random.uniform(0,1,self.X.shape[1])
        val2 = es.compute_naive(cand)
        val1 = es.compute_fast(cand)
        print str(val1) + "==" + str(val2)
        assert(np.abs(val1-val2)<1e-50)
        cand = es._func_sample_locations[npr.randint(0, entropy_search.NUMBER_OF_REPRESENTER_POINTS)]
        val2 = es.compute_naive(cand)
        val1 = es.compute_fast(cand)
        print str(val1) + "==" + str(val2)
        assert(np.abs(val1-val2)<1e-50)

    def xtest_faster_implementation_is_faster(self):
        #N = np.random.randint(50,150)
        N = 50
        entropy_search.NUMBER_OF_CANDIDATE_SAMPLES = 250
        entropy_search.NUMBER_OF_PMIN_SAMPLES = 500
        entropy_search.NUMBER_OF_REPRESENTER_POINTS = 40
        cand = np.random.uniform(0,1,self.X.shape[1])
        es = EntropySearch(self.X, self.y, self.gp)
        t2 = time.time()
        for i in range(0, N):
            es.compute_fast(cand)
        t2 = time.time() - t2
        t1 = time.time()
        for i in range(0,N):
            es.compute_naive(cand)
        t1 = time.time() - t1
        print t1
        print t2
        assert(t2 < t1)

    def xtest_difference_in_pmin_computation(self):
        N = 300
        candidate = np.random.uniform(0,1,self.X.shape[1])
        entropy_search.NUMBER_OF_CANDIDATE_SAMPLES = 150
        entropy_search.NUMBER_OF_PMIN_SAMPLES = 500
        entropy_search.NUMBER_OF_REPRESENTER_POINTS = 40
        es = EntropySearch(self.X, self.y, self.gp)
        mean, L = es._gp.getCholeskyForJointSample(
            np.append(np.array([candidate]), es._func_sample_locations, axis=0))
        #we don't care for the mean of the candidate
        l = np.copy(L[1:,0])
        mean = np.copy(mean[1:])
        L = np.copy(L[1:,1:])
        # print L
        # print mean
        # t1 = time.time()
        # for i in range(0, N):
        #     mean+l*es._omega_cands[0]
        # t1 = time.time() - t1
        # t2 = time.time()
        # for i in range(0, N):
        #     y = es._gp.sample(candidate, es._omega_cands[0])
        #     gp2 = es._gp.copy()
        #     gp2.update(candidate, y)
        # t2 = time.time() - t2
        # print t1
        # print t2
        mean = mean+l*es._omega_cands[0]
        y = es._gp.sample(candidate, es._omega_cands[0])
        gp2 = es._gp.copy()
        gp2.update(candidate, y)

        t1 = time.time()
        for i in range(0, N):
            es._compute_pmin_bins_fast(mean+l*es._omega_cands[0], L)
        t1 = time.time() - t1
        print t1


        t2 = time.time()
        for i in range(0, N):
            y = es._gp.sample(candidate, es._omega_cands[0])
            gp2 = es._gp.copy()
            gp2.update(candidate, y)
            es._compute_pmin_bins(gp2)
        t2 = time.time() - t2
        print t2

        t1 = time.time()
        for i in range(0, N):
            m = np.copy(mean+l*es._omega_cands[0])
            es._compute_pmin_bins_fast(m, L)
        t1 = time.time() - t1
        print t1

    def test_pmin_computation(self):
        '''
        Asserts that the entries of pmin returned by _compute_pmin_bins() sum to 1.
        '''
        es = EntropySearch(self.X, self.y, self.gp)
        pmin = es._compute_pmin_bins(self.gp)
        assert(np.sum(pmin) == 1)

        d = 1
        N = 5
        X = np.zeros([N, d])
        y = np.zeros(N)
        X[0][0] = 0
        y[0] = (X[0] - 0.5) ** 2
        X[1][0] = 1
        y[1] = (X[2] - 0.5) ** 2
        X[2][0] = 0.5
        y[2] = 0
        X[3][0] = 0.5
        y[3] = 0
        X[4][0] = 0.5
        y[4] = 0

        covarname = "Polynomial3"
        cov_func, _ = fetchKernel(covarname)
        noise = 1e-6
        amp2 = 1
        ls = np.ones(getNumberOfParameters(covarname, d))
        mean = 0
        gp = GPModel(X, y, mean, noise, amp2, ls, covarname)
        entropy_search.NUMBER_OF_REPRESENTER_POINTS = N
        es = EntropySearch(X, y, gp)
        es._func_sample_locations = X
        es._Omega[:,2] = 0
        es._Omega[:,3] = es._Omega[:,2]
        es._Omega[:,4] = es._Omega[:,2]
        pmin = es._compute_pmin_bins(gp)
        assert(pmin[2] == pmin[3])
        assert(pmin[2] == pmin[4])
        assert(pmin[0] == 0)
        assert(pmin[1] == 0)
        #pmin should have 3 equally large entries (and 2 with value 0)


    def xtest_pmin_computation_kde(self):
        '''
        Asserts that the entries of pmin returned by _compute_pmin_bins() sum to 1.
        '''
        es = EntropySearch(self.X, self.y, self.gp)
        pmin = es._compute_pmin_kde(self.gp)
        assert(np.abs(np.sum(pmin) - 1) < 1e-15)
        (X, y, gp, es) = _create_setup()
        candidate = np.array([0.5])
        for i in range(0, entropy_search.NUMBER_OF_CANDIDATE_SAMPLES):
            gp2 = es._gp.copy()
            gp2.update(candidate, gp2.sample(candidate, es._omega_cands[i]))
            pmin = es._compute_pmin_kde(gp2)
            #print pmin

    def test_entropy(self):
        '''
        Asserts that the entropy of Pmin is computed as expected. See #_create_setup
        '''
        entropy_search.NUMBER_OF_CANDIDATE_SAMPLES = 2
        #we have two datapoints in 0.2 and 0.8 both with value 0
        (X, y, gp, es) = _create_setup()
        # we need to take a bit extreme candidates to get the result we want
        es._omega_cands[0] = -2
        es._omega_cands[1] = 2

        #let's compute the entropy of Pmin if we'd evaluate in 0.2
        entropy1 = _compute_entropy(es, X[0])

        #now the same for another candidate in 0.5
        entropy2 = _compute_entropy(es, es._func_sample_locations[1])

        #if we evaluate in 0.2 it should tell us nothing about where the minimum is
        #i.e. Pmin(0.2)~0.5 and Pmin(0.5)~0.5
        #whereas if we'd evaluate in 0.5 this would clearly give us Pmin(0.2)=0 or Pmin(0.2)=1 and Pmin(0.5)=1-Pmin(0,2)
        #i.e. an evaluation in 0.5 would reduce the entropy to zero!
        assert(entropy2 <= 1e-15)
        assert(entropy1 >= np.log(1)-1e-15)

    def test_log_proposal_part(self):
        '''
        Asserts that the contribution of the -u^T*Pmin part is correct.
        '''
        #we will draw only ONE sample per candidate
        entropy_search.NUMBER_OF_CANDIDATE_SAMPLES = 1
        (X, y, gp, es) = _create_setup()
        #the samples that ES draws from it's Gaussian process will be below the mean
        es._omega_cands[0] = -1
        #if we evaluate in 0.1 we'd learn that the minimum is in 0.2
        c1 = np.array([0.1])
        #if we evaluate in 0.3 we'd learn the minimum is in 0.5
        c2 = np.array([0.3])
        # However, according to our proposal measure (Expected Improvement) we favour the minimum in 0.5!
        # (we need to turn the sign before compute since we maximize our acquisition functions...)
        prop1 = -es.compute_naive(c1) - _compute_entropy(es, c1)
        prop2 = -es.compute_naive(c2) - _compute_entropy(es, c2)
        assert(prop1 > prop2)


if __name__ == "__main__":
    unittest.main()
