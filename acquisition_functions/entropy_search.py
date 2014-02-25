'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class implements an acquisition function similar to what is proposed in 
"Entropy Search for Information-Efficient Global Optimization" by Hennig and Schuler in 2013.
Instead of using Expectation Propagation we apply sampling.
'''
import numpy as np
import numpy.random as npr
from util import slice_sample
from ..acquisition_functions.expected_improvement import ExpectedImprovement
from ..support.hyper_parameter_sampling import handle_slice_sampler_exception

'''
The number of points used to represent/discretize Pmin.
'''
NUMBER_OF_REPRESENTER_POINTS = 10

'''
The number of independent samples drawn for one candidate to approximate Pmin.  
'''
NUMBER_OF_CAND_SAMPLES = 5

'''
The number of independent joint samples drawn for the representer points.
'''
NUMBER_OF_PMIN_SAMPLES = 20

class EntropySearch(object):
    def __init__(self, comp, vals, gp, cost_gp=None):
        '''
        Default constructor.
        '''
        self._Omega = np.random.normal(0, 1, (NUMBER_OF_CAND_SAMPLES,
                                              NUMBER_OF_PMIN_SAMPLES,
                                              NUMBER_OF_REPRESENTER_POINTS+1))
        self._gp = gp
        self._cost_gp = cost_gp
        starting_point = comp[np.argmin(vals)]
        self._ei = ExpectedImprovement(comp, vals, gp, cost_gp)
        self._representers = np.empty([NUMBER_OF_REPRESENTER_POINTS+1,starting_point.shape[0]])
        self._representers[1:] = self.sample_representer_points(starting_point,
                                                                           self._log_proposal_measure, 
                                                                           NUMBER_OF_REPRESENTER_POINTS)

    #TODO: Is that correct? Do we not have to normalize EI?
    def _log_proposal_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf
        ei = self._ei.compute(x)
        return np.log(ei + 1e-10)

    def sample_representer_points(self, starting_point, log_proposal_measure, number_of_points):
        '''
        Samples representer points for discretization of Pmin.
        Args:
            starting_point: The point where to start the sampling.
            log_proposal_measure: A function that measures in log-scale how suitable a point is to represent Pmin. 
            number_of_points: The number of samples to draw.
        Returns:
            a numpy array containing the desired number of samples
        '''
        #TODO: BUGFIX
        representer_points = np.zeros([number_of_points, starting_point.shape[0]])

        chain_length = 20 * starting_point.shape[0]
        #TODO: burnin?
        for i in range(0, number_of_points):
            #this for loop ensures better mixing
            for c in range(0, chain_length):
                try:
                    starting_point = slice_sample(starting_point, log_proposal_measure)
                except Exception as e:
                    starting_point = handle_slice_sampler_exception(e, starting_point, log_proposal_measure)

            representer_points[i] = starting_point

        return representer_points

    def compute(self, candidate, compute_gradient = False):

        if compute_gradient:
            raise NotImplementedError("computing gradients not supported by this acquisition function")

        loss = 0
        log_proposal_vals = np.zeros(NUMBER_OF_REPRESENTER_POINTS)

        for o in self._Omega:
            pmin = self._compute_pmin_bins(self._gp, candidate, o)
            entropy_pmin = -np.dot(pmin, np.log(pmin + 1e-10))
            #TODO:is it necessary to recompute the proposal measure values of the representer points using the GP copy?
            for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
                log_proposal_vals[i] = self._log_proposal_measure(self._representers[i])

            log_proposal = np.dot(log_proposal_vals, pmin)

            kl_divergence = entropy_pmin - log_proposal

            loss = loss + kl_divergence
        #in the paper the acquisition function is minimized but we maximize

        return -loss

    def _compute_pmin_bins(self, gp, candidate, omega):
        '''
        Computes a discrete belief over Pmin given a Gaussian process using bin method. Leaves
        the Gaussian process unchanged.
        Args:
            gp: the Gaussian process
            candidate: the current candidate
            omega: a sample of the standard normal
        Returns:
            a numpy array with a probability for each representer point
        '''
        pmin = np.zeros(NUMBER_OF_REPRESENTER_POINTS)
        self._representers[0] = candidate
        for o in omega:
            vals = gp.drawJointSample(self._representers, o)
            mins = np.where(vals == vals.min())
            number_of_mins = len(mins)
            #mins is a tuple of single valued array
            for m in mins:
                #have to extract the value of the array
                pmin[m[0]-1] += 1/(number_of_mins)
        pmin = pmin / NUMBER_OF_PMIN_SAMPLES
        return pmin

    def _compute_pmin_kde(self, gp):
        #FIXME: INCORRECT
        '''
        Computes a discrete belief over Pmin given a Gaussian process using kernel density estimator.
        Args:
            gp: the Gaussian process
        Returns:
            a numpy array with a probability for each representer point
        '''
        pmin = np.zeros(len(self._representers))
        for s in xrange(0,NUMBER_OF_PMIN_SAMPLES):
            omega2 = npr.normal(0, 1, NUMBER_OF_REPRESENTER_POINTS) #1-dimensional samples
            vals = gp.drawJointSample(self._representers, omega2)
            mins = np.where(vals == vals.min())
            number_of_mins = len(mins)
            for i in range(0, NUMBER_OF_REPRESENTER_POINTS):
                for m in mins:
                    covar = gp._compute_covariance(np.array([self._representers[i]]), np.array([self._representers[m[0]]]))
                    covar = covar / number_of_mins
                    covar = covar / gp._amp2
                    pmin[i]+=covar
        pmin = pmin / NUMBER_OF_PMIN_SAMPLES
        return pmin

