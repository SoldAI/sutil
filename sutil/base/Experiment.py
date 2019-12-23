# -*- coding: utf-8 -*-

class Experiment:
    
    def __init__(self, data, models = None, train = 0.8, validation=0.2, k_folds = 3):
        self.models = models if models is not None else {}
        self.models_index = {}
        self.train_p = train
        self.validation_p = validation
        self.train, self.validation, self.test = None, None, None
        self.splitData(data)
    
    def splitData(self, data):
        if self.train is None:
            if self.validation_p > 0:
                self.train, self.validation, self.test = data.split(self.train_p, self.validation_p)
            else:
                if self.train_p >= 1:
                    self.train = data
                else:
                    self.train, self.test = data.split(self.train_p, self.validation_p)
        return True
    
    def addModel(self, model, modelClass = None, name = None):
        index = len(self.models)
        name = name if name else "Modelo " + str(len(self.models) + 1)
        self.models[name] = model
        self.models_index[name] = index
        
    def model(self, i = None, name = None):
        if i:
            return self.models[self.models_index[i]]
        else:
            if name:
                return self.models[name]
        return None
    
    def run(self, plot = False):
        for model_name, model in self.models.items():
            print("Training model " + model_name)
            model.trainModel(self.train)
            print("Training score")
            model.score(self.train.X, self.train.y)
            if self.validation is not None:
                print("Validation score")
                model.score(self.validation.X, self.validation.y)
            if self.test is not None:
                print("Test score")
                model.score(self.test.X, self.test.y)
            if plot:
                model.roc.plot()

