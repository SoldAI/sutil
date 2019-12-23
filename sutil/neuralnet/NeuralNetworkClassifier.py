import numpy as np
from sutil.models.Model import Model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class NeuralNetworkClassifier(Model):

    def __init__(self, layers, **kwargs):
        self.initialize(layers, kwargs)
    
    def initialize(self, layers, kwargs):
        self.num_labels = layers[-1]
        self.name = kwargs['name'] if 'name' in  kwargs.keys() else 'Neural Network Model'
        self.solver = kwargs['solver'] if 'solver' in  kwargs.keys() else 'adam'
        self.alpha = kwargs['alpha'] if 'alpha' in  kwargs.keys() else 0.001
        self.activation = kwargs['activation'] if 'activation' in  kwargs.keys() else 'relu'
        self.clf = MLPClassifier(solver=self.solver, activation=self.activation, alpha=self.alpha, hidden_layer_sizes=tuple(layers), random_state=1)
    
    def trainModel(self, data):
        print(data.shape)
        self.clf.fit(data.getBiasedX(), data.y.reshape(data.m, ))
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def searchParameters(self, data, param_grid = None, layers_range = (1, 10)):
        layers = []
        for i in range(layers_range[0], layers_range[1]):
            t = (data.n, i, len(data.labels))
            layers.append(t)
        if not param_grid:
            param_grid = [
                    {'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                     'solver' : ['lbfgs', 'sgd', 'adam'],
                     'hidden_layer_sizes': layers}
                    ]
        gs = GridSearchCV(MLPClassifier(), param_grid, cv=3, scoring='accuracy')
        gs.fit(data.X,data.y)
        print("Best parameters set found on development set:")
        print(gs.best_params_)
        best_layer = gs.best_params_['hidden_layer_sizes']
        self.initialize(best_layer, gs.best_params_)