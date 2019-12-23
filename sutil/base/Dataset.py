# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime

class Dataset(object):

    def __init__(self, X, y, xlabel='x', ylabel="y", legend=["y=1", "y=0"], title="Graph"):
        self.initialize(X, y, xlabel, ylabel, legend, title)
    
    @classmethod
    def fromDataFile(cls, filename, delimiter):
        data = np.loadtxt(filename, delimiter=delimiter)
        X = data[:, 0:-1]
        y = data[:, -1]
        return cls(X, y)

    def initialize(self, X, y, xlabel='x', ylabel="y", legend=["y=1", "y=0"], title="Graph"):
        self.X = X
        self.y = y
        self.shape = (self.X.shape, self.y.shape)
        self.m, self.n = self.X.shape
        self.size = self.m
        self.y.shape = (self.m, 1)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xsize = len(X[0])
        self.ysize = len(y)
        self.title = title
        self.legend = legend
        list_set = {*self.y.flatten().tolist()}
        self.labels = (list(list_set)) 

    def loadData(self, datafile, delimiter):
        data = np.loadtxt(datafile, delimiter)
        self.X = data[:, 0:-1]
        self.y = data[:, -1]
        self.initialize(self.X, self.y)

    def plotData(self, file=None):
        #load the dataset
        pos = np.where(self.y == 1)
        neg = np.where(self.y == 0)
        fig, ax = plt.subplots()
        ax.scatter(self.X[pos, 0], self.X[pos, 1], marker='o', c='b')
        ax.scatter(self.X[neg, 0], self.X[neg, 1], marker='x', c='r')
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        ax.legend(self.legend)
        
        if file:
            plt.savefig(file + '.png')
        plt.show()
            
    def plotDataRegression(self, file=None):
        fig, ax = self.getPlotRegression()
        if file:
            fig.savefig(file + ".png")
        plt.show()
        return fig, ax

    def getPlotRegression(self):
        fig, ax = plt.subplots()
        print(self.X.shape, self.y.shape)
        ax.scatter(self.X, self.y, marker='x', c='r')
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        ax.grid()
        return fig, ax

    def mapFeature(self, x1, x2):
        #Maps the two input features to quadratic features.
        #Returns a new feature array with more features, comprising of
        #1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
        #nputs X1, X2 must be the same size
        x1.shape = (x1.size, 1)
        x2.shape = (x2.size, 1)
        degree = 6
        out = np.ones(shape=(x1[:, 0].size, 1))

        m, n = out.shape

        for i in range(1, degree + 1):
            for j in range(i + 1):
                r = (x1 ** (i - j)) * (x2 ** j)
                out = np.append(out, r, axis=1)
        return out

    def normalizeFeatures(self):
        X_norm = self.X;
        mu = np.zeros((self.n, 1))
        sigma = np.zeros((self.n, 1))
        for i in range(self.n):
            mu[i] = np.mean(self.X[:, i])
            sigma[i] = np.std(self.X[:, i])
            X_norm[:, i] = (self.X[:, i] - mu[i]) / sigma[i]
        self.X = X_norm
        self.mu = mu
        self.sigma = sigma
        return X_norm, mu, sigma
    
    def isNormalized(self):
        return (self.mu is not None and self.sigma is not None)
    
    def normalizeExample(self, example):
        assert(self.isNormalized() == True, 'The normalization process has not been done')
        x_norm = example;
        for i in range(len(x_norm)):
            x_norm[:, i] = (self.x_norm[:, i] - self.mu[i]) / self.sigma[i]
        return x_norm
    
    #Returns the test, validation and test datasets
    def split(self, train = 0.8, validation = 0.2):
        #Split train and test
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.y, test_size = 1 - train)
        if validation > 0:
            X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = validation)
            return self.clone(X_train, Y_train), self.clone(X_validation, Y_validation), self.clone(X_test, Y_test)
        return self.clone(X_train, Y_train), self.clone(X_test, Y_test)
    
    def save(self, filename = None):
        if not filename:
            generation_time = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
            filename = "model_" + generation_time
        filename += ".pickle"
        file = open(filename, 'wb')
        pickle.dump(self, file)
        
    def sample(self, percentage = None, examples = None):
        if not percentage:
            if not examples:
                return self
            else:
                percentage = max(0, examples/self.m)
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.y, test_size = percentage)
        return Dataset(X_test, Y_test)
    
    @staticmethod
    def load(filename):
        file = open(filename + '.pickle', 'rb')
        return pickle.load(file)
    
    def getShape(self):
        return self.X.shape, self.y.shape
    
    def getBiasedX(self):
        return np.hstack((np.ones((len(self.X), 1)), self.X))
    
    def clone(self, X, y):
        return  Dataset(X, y, self.xlabel, self.ylabel, self.legend, self.title)