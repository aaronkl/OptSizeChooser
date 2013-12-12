'''
Created on 12.12.2013

@author: Simon Bartels

This class wraps the hyper-parameter sampling methods for the Gaussian processes of spear mint.
'''

import numpy as np
import scipy.linalg as spla
import util

'''
Global constants.
'''
NOISE_SCALE = 0.1  # horseshoe prior
AMP2_SCALE  = 1    # zero-mean log normal prior
MAX_LS      = 10    # top-hat prior on length scales

def _sample_mean_amp_noise(comp, vals, cov_func, start_point, ls):
    noise = 1e-3
    #if we get a start point that consists only of two variables that means we don't care for the noise
    noiseless = (start_point.shape[0] == 2)
    
    def logprob(hypers):
        mean = hypers[0]
        amp2 = hypers[1]
        if not noiseless:
            noise = hypers[2]

        # This is pretty hacky, but keeps things sane.
        if mean > np.max(vals) or mean < np.min(vals):
            return -np.inf

        if amp2 < 0 or noise < 0:
            return -np.inf
        
        cov = (amp2 * (cov_func(ls, comp, None) + 
            1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), vals - mean)
        lp = -np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve)
        if not noiseless:
            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (NOISE_SCALE / noise) ** 2))

        # Roll in amplitude lognormal prior
        lp -= 0.5 * (np.log(amp2) / AMP2_SCALE) ** 2

        return lp
    return util.slice_sample(start_point, logprob, compwise=False)

def _sample_ls(comp, vals, cov_func, start_point, mean, amp2, noise):
    def logprob(ls):
        if np.any(ls < 0) or np.any(ls > MAX_LS):
            return -np.inf

        cov = (amp2 * (cov_func(ls, comp, None) + 
            1e-6 * np.eye(comp.shape[0])) + noise * np.eye(comp.shape[0]))
        chol = spla.cholesky(cov, lower=True)
        solve = spla.cho_solve((chol, True), vals - mean)

        lp = (-np.sum(np.log(np.diag(chol))) - 0.5 * np.dot(vals - mean, solve))
        return lp

    return util.slice_sample(start_point, logprob, compwise=True)


def sample_hyperparameters(mcmc_iters, noiseless, input_points, func_values, cov_func, noise, amp2, ls):
    '''
    Samples hyper parameters for Gaussian processes.
    Args:
        mcmc_iters: the number of hyper-parameter samples required
        noiseless: the modeled function is noiseless
        input_points: all the points that have been evaluated so far
        func_values: the corresponding observed function values
        cov_func: the covariance function the Gaussian process uses
        noise: a starting value for the noise
        amp2: a starting value for the amplitude
        ls: an array of starting values for the length scales (size has to be the dimension of the input points)
    Returns:
        a list of hyper-parameter tuples
        the tuples are of the form (mean, noise, amplitude, [length-scales])
    '''
    mean = np.mean(func_values)
    hyper_samples = []
    # sample hyper parameters
    for i in xrange(0, mcmc_iters ):
        if noiseless:
            [mean, amp2] = _sample_mean_amp_noise(input_points, func_values, cov_func, np.array([mean, amp2]), ls )
            noise = 1e-3
        else:
            [mean, amp2, noise] =_sample_mean_amp_noise(input_points, func_values, cov_func, np.array([mean, amp2, noise]), ls)
        ls = _sample_ls(input_points, func_values, cov_func, ls, mean, amp2, noise)
        #This is the order as expected
        hyper_samples.append((mean, noise, amp2, ls))
    return hyper_samples