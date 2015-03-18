# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:57:18 2015

@author: p_cohen
"""
from sklearn.ensemble import RandomForestClassifier
class tggBaselineClassifier:
    """ 
    Written by: Peter Cohen
    This class is used to create a baseline predictive model for a
    classification problem.

    
    """
    
    def __init__(self,
                 fitTime):
        self.fitTime = fitTime
        
    def describe(self, X, y):
    def clean(self, X, y):
        """Prepare data (X, y for modeling
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. 
            
        """
        
    
    def model(self, X, y):
    def predict(self, X, y):
    def evaluate(self, y, target, prediction)