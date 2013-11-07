'''
Created on Oct 24, 2013

@author: Simon Bartels
'''

class Model():
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        raise NotImplementedError("Abstract Method!")
        
    
    def predict(self, Xstar, variance=False):
        '''
        Returns a vector of predictions for the given input. If variance is
        true it returns a tuple of vectors where the second entry is the vector
        of variances.
        '''
        raise NotImplementedError("Abstract Method!")
    
    def predictVariance(self, Xstar):
        '''
        Returns a tuple of vectors where the first entry is the vector of
        mean predictions and the second entry is the vector
        of variances.
        '''
        return self.predict(Xstar, True)
    
    def getGradients(self, xstar):
        '''
        Args:
            xstar: A SINGLE vector!
        Returns a tuple of vectors. The first vector contains the gradients
        of the mean. The second vector contains the gradients of the variance.
        '''
        raise NotImplementedError("Abstract Method!")
    
    def getNoise(self):
        '''
        Returns the amount of the noise the model guesses.
        Returns:
            double value >= 0
        '''
        raise NotImplementedError("Abstract Method!")
    
    def optimize(self):
        '''
        Triggers the hyper-parameter optimization of this model.
        '''
        raise NotImplementedError("Abstract Method!")