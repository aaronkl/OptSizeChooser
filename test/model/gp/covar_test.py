'''
Created on 16.01.2014

@author: Simon Bartels

Tests certain abilities of some of the covariance functions.
'''
import unittest
from test.abstract_test import AbstractTest, d, scale
import numpy as np
import scipy.linalg as spla
import numpy.random as npr
from model.gp.gp_model import Polynomial3, grad_Polynomial3, getNumberOfParameters, BigData, GPModel
from support.hyper_parameter_sampling import sample_hyperparameters


class Test(AbstractTest):

    def testPolynomial3(self):
        '''
        Tests the implemented covariance functions.
        '''
        cov_func = Polynomial3
        ls = npr.rand(getNumberOfParameters('Polynomial3', d))
        M1 = cov_func(ls, self.X, self.X)
        #M1 = M ** 3
        #print M1[3][5]
        #print M[3][5] ** 3
        M1=M1#+1e-6*np.eye(self.X.shape[0])
        M2 = np.zeros([self.X.shape[0], self.X.shape[0]])
        for i in range(0, self.X.shape[0]):
            for j in range(0, self.X.shape[0]):
                M2[i][j] = cov_func(ls, self.X[i],self.X[j])
                assert(abs(M1[i][j] -  M2[i][j]) < 1e-5)
        #try:
        spla.cholesky(M1+1e-6*np.eye(self.X.shape[0]), lower=True)
        
        
        xstar = scale * npr.randn(1,d)
        dfdx = grad_Polynomial3(ls, xstar)[0]
        f = lambda x: cov_func(ls, x)
        #TODO: implement correct gradient
        self.assert_first_order_gradient_approximation(f, xstar, dfdx, 1e-6)
        
    #    except _:
            
        #spla.cholesky(M2+1e-6*np.eye(M1.shape[0]), lower=True)
        
        
    def testBigDataKernel(self):
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
            assert(g2 <= g1)
            assert(p2 <= p1)
            
def _makeObservations(dimension, ground_truth):
    N = npr.randint(dimension**4,dimension**7)
    X = npr.random((N,dimension))
    for i in range(0, X.shape[0]):
        #make it harder to obtain samples with full data size
        #=> sample from [0,1) but squared!
        X[i][0] = (npr.random())**2
    y = ground_truth(X)
    return (X,y)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()