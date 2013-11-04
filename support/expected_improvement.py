import numpy        as np
import scipy.stats    as sps

def expected_improvement(model, incumbent, Xstar, gradient=False):
        '''
        If gradient is False this method computes
        the Expected Improvement values for a matrix of input vectors.
        If gradient is True Xstar MUST be a matrix with a single column! In that case
        the return value is a tuple with the value of the EI and its gradient.        
        '''
        #This part works for Xstar as a matrix as well as only a vector
        (func_m, func_v) = model.predictVariance(Xstar)
        # Expected improvement
        func_s = np.sqrt(func_v)
        u = (incumbent - func_m) / func_s
        ncdf = sps.norm.cdf(u)
        npdf = sps.norm.pdf(u)
        v = u * ncdf + npdf
        # this is now a vector containing the EI for each x in X*
        ei = func_s * v
        if not gradient:
            return ei
        #else
        # we can assume that Xstar is a single vector
        # compute gradients of mean and variance
        xstar = Xstar[0]
        (mg, vg) = model.getGradients(xstar)
        sg = 0.5 * vg #we want the gradient of s(x) not of s^2(x)
        grad = v * sg + ncdf * (mg - sg * u)
        return (ei, grad)
