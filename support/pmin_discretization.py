'''
Created on Oct 31, 2013

This file provides utility methods to sample representer/discretizations points for the probability 
distribution Pmin (see for example acquisition_functions.entropy_search).
'''

import util
import numpy as np

def sample_representer_points(starting_point, log_proposal_measure, number_of_points):
    '''
    Samples representer points for discretization of Pmin.
    Args:
        starting_point: The point where to start the sampling.
        log_proposal_measure: A function that measures in log-scale how suitable a point is to represent Pmin. 
        number_of_points: The number of samples to draw.
    Returns:
        A tuple that consists of 
            a numpy array containing the desired number of samples
            a numpy array with the corresponding values of the log proposal measure.
    '''
    representer_points = np.zeros([number_of_points,starting_point.shape[0]])
    log_values = np.zeros(number_of_points)
    for i in range(0,number_of_points):
        representer_points[i] = util.slice_sample(starting_point, log_proposal_measure)
        starting_point = representer_points[i]
        log_values[i] = log_proposal_measure(starting_point)
    return (representer_points, log_values)