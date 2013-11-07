'''
Created on Oct 24, 2013

@author: Simon Bartels
'''
from model.model_factory import ModelFactory
from model.gp import gp_model
import numpy as np


class GPModelFactory(ModelFactory):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def create(self, X, y):
        ls = np.ones(X.shape[1])
        amp2 = 1.0
        noise = 1e-3
        mean = np.mean(y)
        return gp_model.GPModel(X, y, mean, noise, amp2, ls, 'ARDSE')
        