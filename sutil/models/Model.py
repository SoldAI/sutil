# -*- coding: utf-8 -*-
import sutil.base.Dataset as Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sutil.metrics.ModelROC import ModelROC
import numpy as np

class Model(object):
    
    def __init__(self, name = 'Model'):
        self.name = name
    
    def fit(self, X = None, y = None):
        data = Dataset(X, y)
        self.trainModel(data, 100)   
        
    def predict(self, X):
        return X
    
    def score(self, X, y, bias = True):
        if bias:
            X = np.hstack((np.ones((len(X), 1)), X))
        predictions = self.predict(X)
        self.accuracy = accuracy_score(y, predictions, normalize=False)
        self.recall = recall_score(y, predictions)
        self.precission = precision_score(y, predictions)
        self.f1 = f1_score(y, predictions)
        self.roc = ModelROC(self, X, y, self.name)
