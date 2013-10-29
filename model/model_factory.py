'''
Created on Oct 24, 2013

@author: Simon Bartels
'''

class ModelFactory():
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        raise NotImplementedError("Abstract Method!")
        
    def create(self, X, Y):
        '''
        Creates a model.
        '''
        raise NotImplementedError("Abstract Method!")