'''
Created on Oct 24, 2013

@author: Simon Bartels
'''
from model.model_factory import ModelFactory
from model.gp import gp_model


class GPModelFactory(ModelFactory):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        pass
        
    def create(self, X, y):
        return gp_model.GPModel(X, y, "ARDSE")