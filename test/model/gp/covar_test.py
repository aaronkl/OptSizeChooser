'''
Created on 16.01.2014

@author: Simon Bartels

Tests certain abilities of some of the covariance functions.
'''
import unittest
from ...abstract_test import AbstractTest, d, scale
import numpy as np
import scipy.linalg as spla
import numpy.random as npr
from ....gp_model import Polynomial3, grad_Polynomial3, getNumberOfParameters, \
BigData, grad_BigData, GPModel, Normalized_Polynomial3, grad_Normalized_Polynomial3, \
CostKernel, grad_CostKernel, CostKernel2, grad_CostKernel2
from ....support.hyper_parameter_sampling import sample_hyperparameters
import gp

def _invert_sign(grad_cov):
    '''
    Returns a copy of the given gradient covariance function that differs by sign.
    (Necessary because the signs differ with the finite differences approximation)
    '''
    def inv_grad_cov(ls, X, x=None): return -grad_cov(ls, X, x)
    return inv_grad_cov

class Test(AbstractTest):

    def _testCovar(self, cov_name, cov_func, grad_cov_func):
        '''
        Basic method for testing. Tests that cholesky decomposition is possible and that gradient is approximately
        correct.
        Args:
            cov_name: the name of the covariance function
            cov_func: the covariance function k(x,y) to be tested
            grad_cov_func: function that computes dk(x,y)/dx
        '''
        ls = npr.rand(getNumberOfParameters(cov_name, d))
        M1 = cov_func(ls, self.X, self.X)
        M2 = np.zeros([self.X.shape[0], self.X.shape[0]])
        for i in range(0, self.X.shape[0]):
            for j in range(0, self.X.shape[0]):
                #M2[i][j] = cov_func(ls, self.X[i],self.X[j])
                c = cov_func(ls, np.array([self.X[i]]),np.array([self.X[j]]))
                #print c
                M2[i][j] = c
                assert(abs(M1[i][j] -  M2[i][j]) < 1e-5)
        #try:
        spla.cholesky(M1+1e-6*np.eye(self.X.shape[0]), lower=True)
        
        #test with argument x2=none
        xstar = scale * npr.randn(1,d)  
        dfdx = grad_cov_func(ls, xstar)
        f = lambda x: cov_func(ls, x)
        self.assert_first_order_gradient_approximation(f, xstar, dfdx, 1e-13)
        
        #test with first argument being only a vector
        x1 = scale * npr.randn(1,d)
        dfdx = grad_cov_func(ls, x1, xstar)
        f = lambda x: cov_func(ls, x1, x)
        self.assert_first_order_gradient_approximation(f, xstar, dfdx, 1e-13)
        
        #test with first argument being a matrix
        dfdx = grad_cov_func(ls, self.X, xstar)
        f = lambda x: cov_func(ls, self.X, x).T #this way we get an appropriate vector
        self.assert_first_order_gradient_approximation(f, xstar, dfdx, 1e-13)
        
    def testPolynomial3(self):
        self._testCovar('Polynomial3', Polynomial3, _invert_sign(grad_Polynomial3))
        
#     def testNormalizedPolynomial3(self):
#         self._testCovar('Normalized_Polynomial3', Normalized_Polynomial3, _invert_sign(grad_Normalized_Polynomial3))
#           
    def testARDSE(self):
        name = "ARDSE"
        cov = getattr(gp, name)
        grad_cov = getattr(gp, 'grad_' + name)
        self._testCovar(name, cov, _invert_sign(grad_cov))
          
    def testMatern52(self):
        name = 'Matern52'
        cov = getattr(gp, name)
        grad_cov = getattr(gp, 'grad_' + name)
        self._testCovar(name, cov,  _invert_sign(grad_cov))
          
    def testBigData(self):
        self._testCovar('BigData', BigData,  _invert_sign(grad_BigData))
         
#     def testBigDataLegacy(self):
#         self._testCovar('BigData', BigData_legacy, _invert_sign(grad_BigData_legacy))
        
    def testCostKernel(self):
        self._testCovar('CostKernel', CostKernel,  _invert_sign(grad_CostKernel))
    
    def testCostKernel2(self):
        self._testCovar('CostKernel2', CostKernel2,  _invert_sign(grad_CostKernel2))
        
    
          
    def testBigDataKernelIsMonotone(self):
        '''
        This function tests that Gaussian processes using this kernel predict
        lower function values the higher the first input argument.
        '''
        f = lambda x: 1-x[1]*x[0]+x[1]/2 #1+np.tanh(2*(x[1]-x[0]))#1+2*np.sin(2*np.pi*(x[0]+x[1]))#(1+(0.5-x[0]+x[1])**2)#np.sqrt(np.abs(x[1]-x[0]))#6 * (x[1] * np.sin(x[0]) + (1-x[1]) * np.cos(x[0]))#20*(np.sqrt(x[1])-x[0])**2 #np.abs(np.sin((x[1]+x[0])/1e-4))
        def ground_truth(X):
            y = np.zeros(X.shape[0])
            d = X.shape[1]
            for i in range(0, X.shape[0]):
                y[i] = f(X[i])#/(np.sqrt(48*X[i][0]+1))
            return y
        dimension = 2
        (X,y) = _makeObservations(dimension, ground_truth)
        covarname = 'BigData'
        cov_func = BigData
        noise = 1e-6
        amp2 = np.std(y)+1e-4
        ls = np.ones(getNumberOfParameters(covarname, dimension))
        noiseless = bool(npr.randint(2))
        mcmc_iters = npr.randint(10,20)
        gp_params = sample_hyperparameters(mcmc_iters, noiseless, X, y, cov_func, noise, amp2, ls)[mcmc_iters-1]
        gp = GPModel(X, y, gp_params[0], gp_params[1], gp_params[2], gp_params[3], covarname)
        #import support.Visualizer as vis
        #vis.plot2DFunction(ground_truth)
        #vis.plot2DFunction(lambda x: gp.predict(x))
        n = npr.randint(5, 15)
        Xstar = npr.rand(n, dimension)
        for x in range(0, n):
            xstar = np.array([Xstar[x]])
            epsilon = npr.random()*(1-xstar[0][0])
            p1 = gp.predict(xstar)
            g1 = ground_truth(xstar)
            #increase first component, i.e. more data
            xstar[0][0]=xstar[0][0]+epsilon
            p2 = gp.predict(xstar)
            g2 = ground_truth(xstar)
            if not g2 <= g1:
                print ("Ground truth is not monotone!")
                assert(g2<=g1)
            assert(p2 <= p1)
            
    def testProductKernelVsOriginalBigDataImpl(self):
        '''
        Compares the original implementation of the big data kernel against the
        more generic implementation as product kernel. 
        '''
        
        ls = npr.rand(getNumberOfParameters('BigData', d))
        xstar = scale * npr.randn(1,d)
        x2 = scale * npr.randn(1,d)
        assert(np.allclose(BigData_legacy(ls, xstar), BigData(ls, xstar)))
        assert(np.allclose(BigData_legacy(ls, x2, xstar), BigData(ls, x2, xstar)))
        assert(np.allclose(BigData_legacy(ls, self.X, xstar), BigData(ls, self.X, xstar)))
        assert(np.allclose(BigData_legacy(ls, self.X), BigData(ls, self.X)))
        
        assert(np.allclose(grad_BigData_legacy(ls, xstar), grad_BigData(ls, xstar)))
        assert(np.allclose(grad_BigData_legacy(ls, x2, xstar), grad_BigData(ls, x2, xstar)))
        assert(np.allclose(grad_BigData_legacy(ls, self.X, xstar), grad_BigData(ls, self.X, xstar)))
            
def _makeObservations(dimension, ground_truth):
    N = npr.randint(dimension**4,dimension**7)
    X = npr.random((N,dimension))
    for i in range(0, X.shape[0]):
        #make it harder to obtain samples with full data size
        #=> sample from [0,1) but squared!
        X[i][0] = (npr.random())**2
    y = ground_truth(X)
    return (X,y)

####Legacy code of the bigdata kernel.
def _bigData_raw(ls, x1, x2=None, value=True, grad=False):
    k1x2 = None
    k2x2 = None
    #separate input vector(s) after first dimension
    k1x1 = x1[:,:1] #get first entry of each vector
    k2x1 = x1[:,1:] #get the rest
    if not(x2 is None):
        k1x2 = x2[:,:1]
        k2x2 = x2[:,1:]
    
    if not grad:
        #only the value is of interest
        k1 = Polynomial3(ls[:1], k1x1, k1x2)
        k2 = gp.Matern52(ls[1:], k2x1, k2x2)
        k = np.array([k1[i]*k2[i] for i in range(0, x1.shape[0])])
        return k
    else:
        (k1, dk1) = Polynomial3(ls[:1], k1x1, k1x2, grad)
        (k2, dk2) = gp.Matern52(ls[1:], k2x1, k2x2, grad)
        #product rule
        dk = np.array([np.concatenate((dk1[i]*k2[i], k1[i]*dk2[i]), axis=1) for i in range(0, x1.shape[0])])
        if not value:
            #we care only for the gradient
            return dk
        k = np.array([k1[i]*k2[i] for i in range(0, x1.shape[0])])
        return (k,dk)
    
def grad_BigData_legacy(ls,x1,x2=None):
    return _bigData_raw(ls, x1, x2, value=False, grad=True)

def BigData_legacy(ls, x1, x2=None, grad=False):
    return _bigData_raw(ls, x1, x2, True, grad)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()