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
        # this is now a vector containing the EI for each x in X*
        ei = func_s * (u * ncdf + npdf)
        if not gradient:
            return ei
        #else
        # we can assume that Xstar is matrix with a single column
        # compute gradients of mean and variance
        xstar = Xstar[0]
        #TODO: This is actually a bit inefficient, since getGradient computes the variance again
        #which we already have here!
        (mg, vg) = model.getGradients(xstar)
        sg = 0.5 * vg #we want the gradient of s(x) not of s^2(x)
        
        # this is the result after some simplifications
        grad = npdf * sg + ncdf * mg
        
        #TODO: For which reason ever grad deviates from the spear mint grad by a factor of 2!
        #So, since we assume the spear mint implementation as correct (which might be wrong
        #in this case) we divide by 2 here.
        return (ei, grad[0]/2)
