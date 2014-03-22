'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class implements an acquisition function similar to what is proposed in 
"Entropy Search for Information-Efficient Global Optimization" by Hennig and Schuler in 2013.
Instead of using Expectation Propagation we do Monte Carlo integration.
'''
import numpy as np
from util import slice_sample
from ..acquisition_functions.expected_improvement import ExpectedImprovement
from ..support.hyper_parameter_sampling import handle_slice_sampler_exception
from scipy.stats import norm
from sobol_lib import i4_sobol_generate
import time

'''
The number of points used to represent/discretize Pmin (without the candidate).
'''
NUMBER_OF_REPRESENTER_POINTS = 20

'''
The number of independent joint samples drawn for the representer points.
'''
NUMBER_OF_PMIN_SAMPLES = 600

'''
The number of samples per candidate.
'''
NUMBER_OF_CANDIDATE_SAMPLES = 150

def sample_representer_points(starting_point, log_proposal_measure, number_of_points):
    '''
    Samples representer points for discretization of Pmin.
    Args:
        starting_point: The point where to start the sampling.
        log_proposal_measure: A function that measures in log-scale how suitable a point is to represent Pmin.
        number_of_points: The number of samples to draw.
    Returns:
        a numpy array containing the desired number of samples
    '''
    representer_points = np.zeros([number_of_points,starting_point.shape[0]])
    chain_length = 20 * starting_point.shape[0]
    #TODO: burnin?
    for i in range(0,number_of_points):
        #this for loop ensures better mixing
        for c in range(0, chain_length):
            try:
                starting_point = slice_sample(starting_point, log_proposal_measure)
            except Exception as e:
                starting_point = handle_slice_sampler_exception(e, starting_point, log_proposal_measure)
        representer_points[i] = starting_point
    return representer_points

class EntropySearch(object):
    def __init__(self, comp, vals, gp, cost_gp=None):
        '''
        Default constructor.
        '''
        self._gp = gp
        self._cost_gp = cost_gp
        self._ei = ExpectedImprovement(comp, vals, gp, cost_gp)

        #samples for the reprensenter points to compute Pmin
        #self._Omega = np.random.normal(0, 1, (NUMBER_OF_PMIN_SAMPLES,
        #                                      NUMBER_OF_REPRESENTER_POINTS))
        self._Omega = norm.ppf(i4_sobol_generate(NUMBER_OF_REPRESENTER_POINTS, NUMBER_OF_PMIN_SAMPLES+1, 1)[:,1:]).T
        #we skip the first entry since it will yield 0 and ppf(0)=-infty

        #samples for the candidates
        #self._omega_cands = np.random.normal(0, 1, NUMBER_OF_CANDIDATE_SAMPLES)
        #we use stratified sampling for the candidates
        self._omega_cands = norm.ppf(np.linspace(1./(NUMBER_OF_CANDIDATE_SAMPLES+1),
                                           1-1./(NUMBER_OF_CANDIDATE_SAMPLES+1),
                                           NUMBER_OF_CANDIDATE_SAMPLES))

        starting_point = comp[np.argmin(vals)]
        self._func_sample_locations = sample_representer_points(starting_point,self._log_proposal_measure,
                                        NUMBER_OF_REPRESENTER_POINTS)
        self._log_proposal_vals = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        #u(x) is fixed - therefore we can compute it here
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            self._log_proposal_vals[i] = self._log_proposal_measure(self._func_sample_locations[i])

    def _log_proposal_measure(self, x):
        if np.any(x<0) or np.any(x>1):
            return -np.inf
        v = self._ei.compute(x)
        return np.log(v+1e-10)

    def compute_fast(self, candidate, compute_gradient = False):
        if compute_gradient:
            raise NotImplementedError("computing gradients not supported by this acquisition function")
        kl_divergence = 0
        #TODO: the Cholesky for the representer points is always the same
        # it would be faster to do a rank 1 update for each candidate
        mean, L = self._gp.getCholeskyForJointSample(
            np.append(np.array([candidate]), self._func_sample_locations, axis=0))
        l = np.copy(L[1:,0])
        mean = np.copy(mean[1:])
        L = np.copy(L[1:,1:]) #appearantly it's faster to copy
        for i in range(0, NUMBER_OF_CANDIDATE_SAMPLES):
            pmin = self._compute_pmin_bins_fast(mean+l*self._omega_cands[i], L)
            entropy_pmin = -np.dot(pmin, np.log(pmin+1e-50))
            log_proposal = np.dot(self._log_proposal_vals, pmin)
            #division by NUMBER_OF_CANDIDATE_SAMPLES to keep things numerically stable
            kl_divergence += (entropy_pmin - log_proposal)/NUMBER_OF_CANDIDATE_SAMPLES
        #since originally KL is minimized and we maximize our acquisition functions we have to turn the sign
        return -kl_divergence

    def _compute_pmin_bins_fast(self, mean, L):
        pmin = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        for omega in self._Omega:
            vals = mean + np.dot(L, omega)
            mins = np.where(vals == vals.min())[0] #the return value is a tuple
            number_of_mins = len(mins)
            for m in mins:
                pmin[m] += 1./(number_of_mins)
        pmin = pmin / NUMBER_OF_PMIN_SAMPLES
        return pmin

    def compute(self, candidate, compute_gradient = False):
        return self.compute_fast(candidate, compute_gradient)

    def _compute_pmin_bins(self, gp):
        '''
        Computes a discrete belief over Pmin given a Gaussian process using bin method. Leaves
        the Gaussian process unchanged.
        Returns:
            a numpy array with a probability for each representer point
        '''
        pmin = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        mean, L = gp.getCholeskyForJointSample(self._func_sample_locations)
        for omega in self._Omega:
            vals = gp.drawJointSample(mean, L, omega)
            mins = np.where(vals == vals.min())[0] #the return value is a tuple
            number_of_mins = len(mins)
            for m in mins:
                pmin[m] += 1./(number_of_mins)
        pmin = pmin / NUMBER_OF_PMIN_SAMPLES
        return pmin


    def compute_naive(self, candidate, compute_gradient = False):
        if compute_gradient:
            raise NotImplementedError("computing gradients not supported by this acquisition function")
        kl_divergence = 0
        for i in range(0, NUMBER_OF_CANDIDATE_SAMPLES):
            #it does NOT produce better results if we don't reuse samples here
            # the function just looks more wiggly but has the same problem that it drifts if there are not enough
            # representer points
            y = self._gp.sample(candidate, self._omega_cands[i])
            gp2 = self._gp.copy()
            gp2.update(candidate, y)
            pmin = self._compute_pmin_bins(gp2)
            entropy_pmin = -np.dot(pmin, np.log(pmin+1e-50))
            log_proposal = np.dot(self._log_proposal_vals, pmin)
            #division by NUMBER_OF_CANDIDATE_SAMPLES to keep things numerically stable
            kl_divergence += (entropy_pmin - log_proposal)/NUMBER_OF_CANDIDATE_SAMPLES
        #since originally KL is minimized and we maximize our acquisition functions we have to turn the sign
        return -kl_divergence
    def _compute_pmin_kde(self, gp):
        '''
        THIS FUNCTION DOES NOT WORK!

        Computes a discrete belief over Pmin given a Gaussian process using kernel density estimator.
        Args:
            gp: the Gaussian process
        Returns:
            a numpy array with a probability for each representer point
        '''
        #Should anyone consider using this method: move the function below somewhere else and...
        def SE(xx1, xx2):
            r2 = np.maximum(-(np.dot(xx1, 2*xx2.T)
                           - np.sum(xx1*xx1, axis=1)[:,np.newaxis]
                           - np.sum(xx2*xx2, axis=1)[:,np.newaxis].T), 0.0)
            cov = np.exp(-0.5 * r2)
            return cov
        #... this part into the constructor.
        self._normalization_constant = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
            for j in range(0, NUMBER_OF_REPRESENTER_POINTS):
                self._normalization_constant[i] += SE(np.array([self._func_sample_locations[i]]),
                                                      np.array([self._func_sample_locations[j]]))

        pmin = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        for omega in self._Omega:
            vals = gp.drawJointSample(self._func_sample_locations, omega)
            mins = np.where(vals == vals.min())
            number_of_mins = len(mins)
            for m in mins:
                m = m[0]
                for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
                    pmin[i]+=SE(np.array([self._func_sample_locations[i]]), np.array([self._func_sample_locations[m]]))\
                             /self._normalization_constant[m]
                    pmin[i]/=number_of_mins
        pmin = pmin / NUMBER_OF_PMIN_SAMPLES
        return pmin
