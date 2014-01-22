'''
Created on Oct 28, 2013

@author: Simon Bartels, Aaron Klein
'''

import gp
import numpy as np
import scipy.linalg as spla
from ..model import Model
import copy

def _polynomial3_raw(ls, x1, x2=None, value=True, grad=False):
    factor = 1
    if x2 is None:
        #in this case k(x,y) has to be considered of the form k(x)
        #and dk/dx is then 3(x^Tx+c)^2*(2x)
        factor = 2
        x2=x1
    #we may assume the input is a matrix of the form N x D
    #c = np.empty([x1.shape[0],x2.shape[0]])
    #c.fill(ls[0])
    c = ls[0]
    dot = np.dot(x1, x2.T) + c
    if grad:
        #compute dk(x1,x2)/dx2
        #=3 * apply(dot, ^2) o x1
        #dk = factor * 3 * (dot ** 2) * x1
        #this is to get the gradient in spearmint's format
        dk = np.array([(factor * 3 * (dot[i] ** 2)) * np.array([x1[i]]) for i in range(0, x1.shape[0])])
        
        #because the spearmint implementations all invert the signs of their gradients we will do so as well
        dk = -dk        
        if value:
            #we want both: value and gradient
            k = dot ** 3
            return (k, dk)
        #we want just the gradient
        return dk
    else:
        #we want just the kernel value
        k = dot ** 3
        return k

def grad_Polynomial3(ls, x1, x2=None):
    return _polynomial3_raw(ls, x1, x2, value=False, grad=True)

def Polynomial3(ls, x1, x2=None, grad=False):
    return _polynomial3_raw(ls, x1, x2, True, grad)

def Normalized_Polynomial3(ls, x1, x2=None, grad=False):
    #TODO: quick and dirty, refactor!
    compute_grad = False
    if grad:
        (k, dk) = _polynomial3_raw(ls, x1, x2, True, True)
        compute_grad = True
    else:
        k = _polynomial3_raw(ls, x1, x2)
    if x2 is None:
        x2 = x1
        if grad:
            dk = np.array([np.array([np.zeros(x1.shape[1])])])
            compute_grad = False
    for i in range(0, x1.shape[0]):
        if compute_grad:
            #we assume x2 was not none!!!
            sqrt_kxx = np.sqrt(_polynomial3_raw(ls, np.array(x1[i])))
            sqrt_kyy = np.sqrt(_polynomial3_raw(ls, x2))
            kxy = k[i]
            dk[i][0] = 1/sqrt_kxx*(dk[i][0]/sqrt_kyy-0.5*kxy*grad_Polynomial3(ls, x2)[0]/(sqrt_kyy**3))
        for j in range(0, x2.shape[0]):
            ki = _polynomial3_raw(ls, np.array(x1[i]))
            kj = _polynomial3_raw(ls, np.array(x2[j]))
            k[i][j] = k[i][j]/(np.sqrt(ki)*np.sqrt(kj))
    if not grad:
        return k
    return (k,dk)

def grad_Normalized_Polynomial3(ls, x1, x2=None):
    #TODO: quick and dirty, refactor!
    (k, dk) = _polynomial3_raw(ls, x1, x2, True, True)
    if x2 is None:
        return np.array([np.array([np.zeros(x1.shape[1])])])
    for i in range(0, x1.shape[0]):
        #we assume x2 was not none!!!
        sqrt_kxx = np.sqrt(_polynomial3_raw(ls, np.array(x1[i])))
        sqrt_kyy = np.sqrt(_polynomial3_raw(ls, x2))
        kxy = k[i]
        dk[i][0] = 1/sqrt_kxx*(dk[i][0]/sqrt_kyy-0.5*kxy*grad_Polynomial3(ls, x2)[0]/(sqrt_kyy**3))
    return dk

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
        dk = np.array([np.concatenate((dk1[i]*k2[i], k1[i]*dk2[i]), axis=1) for i in range(0, x1.shape[0])])
        if not value:
            #we care only for the gradient
            return dk
        k = np.array([k1[i]*k2[i] for i in range(0, x1.shape[0])])
        return (k,dk)

def grad_BigData(ls,x1,x2=None):
    return _bigData_raw(ls, x1, x2, value=False, grad=True)

def BigData(ls, x1, x2=None, grad=False):
    return _bigData_raw(ls, x1, x2, True, grad)

def getNumberOfParameters(covarname, input_dimension):
    '''
    Returns the number of parameters the kernel has.
    Args:
        covarname: the name of the covariance function
        input_dimension: dimensionality of the input arguments
    Returns:
        the number of parameters
    '''
    try:
        #try to find covariance function in spearmint GP class
        getattr(gp, covarname)
        #then it is just the input dimension
        return input_dimension
    except:
        if covarname == 'Polynomial3':
            return 1
        elif covarname == 'Normalized_Polynomial3':
            return getNumberOfParameters('Polynomial3', input_dimension)
        elif covarname == 'BigData':
            return getNumberOfParameters('Polynomial3', 1)+getNumberOfParameters('Matern52', input_dimension-1)
        else:
            raise NotImplementedError('The given covariance function (' + covarname + 'was not found.')
        


class GPModel(Model):

    def __init__(self, X, y, mean, noise, amp2, ls, covarname="Matern52", cholesky=None, alpha=None, cov_func = None, covar_derivative=None):
        '''
            Constructor
            Args:
            X: The observed inputs.
            y: The corresponding observed values.
            mean: the mean value
            noise: the noise value
            amp: the amplitude
            ls: the length scales (numpy array of length equialent to the dimension of the input points)
            covarname: the name of the covariance function (see spearmint class gp)
            
            The following arguments are only used internally for the copy() method.
            cholesky: if already available the cholesky of the kernel matrix
            alpha: if cholesky is passed this one needs to be set, too. It's L/L/(y-mean).
        '''
        
        self._X = X
        self._y = y
        self._ls = ls
        self._amp2 = amp2
        self._mean = mean
        self._noise = noise
        if cholesky is None:
            try:
                #try to find covariance function in spearmint GP class
                self._cov_func = getattr(gp, covarname)
                self._covar_derivative = getattr(gp, "grad_" + covarname)
            except:
                #try to find covariance funtion in THIS class
                self._cov_func = globals()[covarname]
                self._covar_derivative = globals()["grad_" + covarname]
            self._compute_cholesky()
        else:
            self._cov_func = cov_func
            self._covar_derivative =covar_derivative
            self._L = cholesky
            self._alpha = alpha
        
        
    def _compute_cholesky(self):
        #the Cholesky of the correlation matrix
        K = self._compute_covariance(self._X) + self._noise * np.eye(self._X.shape[0])
        self._L = spla.cholesky(K, lower=True)
        self._alpha = spla.cho_solve((self._L, True), self._y - self._mean)
        
    def predict(self, Xstar, variance=False):
        kXstar = self._compute_covariance(self._X, Xstar)
        func_m = np.dot(kXstar.T, self._alpha) + self._mean
        if not variance:
            return func_m
        
        beta = spla.solve_triangular(self._L, kXstar, lower=True)
        #TODO: change back
        #func_v = self._amp2 * (1 + 1e-6) - np.sum(beta ** 2, axis=0)
        func_v = self._compute_covariance(Xstar) - np.dot(beta.T, beta)
        return (func_m, func_v)

    def predict_vector(self, input_point):
        (func_m, func_v) = self.predict(np.array([input_point]), True)
        return (func_m[0], func_v[0])
    
    def _compute_covariance(self, x1, x2=None):
        if x2 is None:
            return self._amp2 * (self._cov_func(self._ls, x1, None)
                                + 1e-6*np.eye(x1.shape[0]))
        else:
            return self._amp2 * self._cov_func(self._ls, x1, x2)
        
    def getGradients(self, xstar):
        xstar = np.array([xstar])
        # gradient of mean
        #This is what Andrew McHutchon in "Differentiating Gaussian Processes"
        #proposes. 
        #dk = self._amp2 * self._covar_derivative(self._ls, xstar, self._X)[0]
        #grad_m = np.dot(dk.T, self._alpha)
        #The sign of the version above agrees with the first order approximation.
        #The spearmint implementation not.
        #THEREFORE WE TURN THE SIGN OF DK! BEWARE: Also used for grad_v!!!
        # Below is the code how spear mint does it.
        dk = -self._amp2 * self._covar_derivative(self._ls, self._X, xstar)
        dk = np.squeeze(dk)
        grad_m = np.dot(self._alpha.T, dk)
        
        # gradient of variance    
        #intuitively k should be cov(xstar,X) but that gives a row vector!
        k = self._compute_covariance(self._X, xstar)
        #kK = k^T * K^-1
        kK = spla.cho_solve((self._L, True), k)
        #s'(x)=-dk^T kK /s(x). So for the derivative of v(x) in terms of s(x) we have:
        #v(x)=s^2(x) <=> v'(x)=2s(x)*s'(x)
        grad_v = -2 * np.dot(kK.T, dk)
        return (grad_m, grad_v)

    def getNoise(self):
        return self._noise
    
    def getAmplitude(self):
        return self._amp2
        
    def sample(self, x, omega):
        '''
            Gives a stochastic prediction at x
            Parameters In: omega = sample from standard normal distribution 
                            x = the prediction point (vector)
            Parameters Out: y = function value at x
        '''
        x = np.array([x])
        cholsolve = spla.cho_solve((self._L, True), self._compute_covariance(self._X, x))
        cholesky = np.sqrt(self._compute_covariance(x, x) - 
                           np.dot(self._compute_covariance(x, self._X), cholsolve))
        y = self.predict(x,False) + cholesky * omega
        #y is in the form [[value]] (1x1 matrix)
        return y[0][0]
        
    def update(self, x, y):
        '''
            Adds x,y to the observation and creates a new GP 
            Args:
                x: a numpy vector
                y: a single value
        '''
        self._X = np.append(self._X, np.array([x]), axis=0)
        self._y = np.append(self._y, np.array([y]), axis=0)
        #TODO: Use factor update
        self._compute_cholesky()
        
    def drawJointSample(self, Xstar, omega):
        '''
            Draws a joint sample at the given points.
            Args:
                X: a numpy array of points, i.e. a matrix
                omega: a vector of samples from the standard normal distribution, one for each point
            Returns:
                a numpy array (vector)
        '''
        kXstar = self._compute_covariance(self._X, Xstar)
        cholsolve = spla.cho_solve((self._L, True), kXstar)
        Sigma = (self._compute_covariance(Xstar, Xstar) -
                  np.dot(kXstar.T, cholsolve))
        cholesky = spla.cholesky(Sigma + 1e-6*np.eye(Sigma.shape[0]))
        y = self.predict(Xstar,False) + np.dot(cholesky, omega)
        return y
    
    def copy(self):
        '''
        Returns a copy of this object.
        Returns:
            a copy
        '''
        X = copy.copy(self._X)
        y = copy.copy(self._y)
        ls = copy.copy(self._ls)
        L = copy.copy(self._L)
        alpha = self._alpha
        return GPModel(X, y, self._mean, self._noise, self._amp2, ls, None, L, alpha, self._cov_func, self._covar_derivative)