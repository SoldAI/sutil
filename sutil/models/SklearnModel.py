# -*- coding: utf-8 -*-
from sutil.models.Model import Model

class SklearnModel(Model):
    
    def __init__(self, name, external):
        self.name = name
        self.external = external
    
    def trainModel(self, data):
        #Normalize and bias
        self.external.fit(data.getBiasedX(), data.y)
    
    def predict(self, X):
        return self.external.predict(X)
