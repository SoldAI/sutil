# -*- coding: utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import datetime
import random

class Dataset(object):

    def __init__(self, X, y, xlabel='x', ylabel="y", legend=["y=1", "y=0"], title="Graph"):
        self.initialize(X, y, xlabel, ylabel, legend, title)

    @classmethod
    def fromDataFile(cls, filename, delimiter, ysize = 1):
        data = np.loadtxt(filename, delimiter=delimiter)
        X = data[:, 0:-ysize]
        y = data[:, -ysize]
        return cls(X, y)

    @staticmethod
    def load(filename):
        file = open(filename + '.pickle', 'rb')
        return pickle.load(file)

    def initialize(self, X, y, xlabel='x', ylabel="y", legend=None, title="Graph"):
        self.setData(X, y, xlabel, ylabel, legend, title)

    def setData(self, X, y, xlabel='x', ylabel="y", legend=None, title="Graph"):
        self.X = X
        self.y = y
        self.shape = (self.X.shape, self.y.shape)
        self.m, self.n = self.X.shape
        self.size = self.m
        self.y.shape = (self.m, 1)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xsize = len(X[0])
        self.ysize = len(y[0])
        self.title = title
        list_set = {*self.y.flatten().tolist()}
        self.labels = (list(list_set))
        if not legend:
            self.legend = []
            for l in self.labels:
                self.legend.append("y=" + str(l))
        else:
            self.legend = legend

    def loadData(self, datafile, delimiter):
        data = np.loadtxt(datafile, delimiter)
        self.X = data[:, 0:-1]
        self.y = data[:, -1]
        self.initialize(self.X, self.y)

    def plotData(self, file=None, d3=False, show_legend = True):
        #The plotted data is assuming 2 dimenssions, check what happened if it's more how to make the projection into 2 dimensions
        classes = self.labels
        if d3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
        identifiers = {}
        used = []
        x_graph = self.X
        if len(self.X) > 2:
            pca_analyzer = PCA(n_components=2)
            if d3:
                pca_analyzer = PCA(n_components=3)
            x_graph = pca_analyzer.fit_transform(self.X)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        markers = ['o', 'x', 'v', '^', 's', 'p', 'P', '*', 'h', '1', '2', '3', '4']
        for c in classes:
            if len(used) == len(colors) * len(markers):
                used = []
            assigned = False
            c_index, m_index = 0, 0
            while not assigned:
                c_index, m_index = self.getRandomIdentifier(colors, markers)
                string = colors[c_index] + markers[m_index]
                if string not in used:
                    used.append(string)
                    assigned = True
            index = np.where(self.y == c)
            identifiers[c] = {'color': colors[c_index], 'marker': markers[m_index], "legend": "y=" + str(c)}
            if d3:
                ax.scatter(x_graph[index, 0], x_graph[index, 1], x_graph[index, 2], marker=markers[m_index], c=colors[c_index])
            else:
                ax.scatter(x_graph[index, 0], x_graph[index, 1], marker=markers[m_index], c=colors[c_index])
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        if show_legend:
            ax.legend(self.legend)

        if file:
            plt.savefig(file + '.png')
        plt.show()

    def getRandomIdentifier(self, colors, markers):
        color = random.randint(0, len(colors)-1)
        marker = random.randint(0, len(markers)-1)
        return (color, marker)

    def plotDataRegression(self, file=None, randomize_marker = False):
        fig, ax = self.getPlotRegression(randomize_marker)
        if file:
            fig.savefig(file + ".png")
        plt.show()
        return fig, ax

    def getPlotRegression(self, randomize_marker = False):
        fig, ax = plt.subplots()
        print(self.X.shape, self.y.shape)
        if randomize_marker:
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            markers = ['o', 'x', 'v', '^', 's', 'p', 'P', '*', 'h', '1', '2', '3', '4']
            c_index, m_index = self.getRandomIdentifier(colors, markers)
            ax.scatter(self.X, self.y, marker=markers[m_index], c=colors[c_index])
        else:
            ax.scatter(self.X, self.y, marker='x', c='r')
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel, title=self.title)
        ax.legend(self.legend)
        ax.grid()
        return fig, ax

    def normalizeFeatures(self):
        X_norm = self.X;
        mu = np.zeros((self.n, 1))
        sigma = np.zeros((self.n, 1))
        for i in range(self.n):
            mu[i] = np.mean(self.X[:, i])
            sigma[i] = np.std(self.X[:, i])
            if sigma[i] == 0:
                X_norm[:, i] = (self.X[:, i] - mu[i])
            else:
                X_norm[:, i] = (self.X[:, i] - mu[i]) / sigma[i]
        self.X = X_norm
        self.mu = mu
        self.sigma = sigma
        return X_norm, mu, sigma

    def isNormalized(self):
        return (self.mu is not None and self.sigma is not None)

    def normalizeExample(self, example):
        if not self.isNormalized():
            return example
        x_norm = example
        for i in range(len(x_norm)):
            if self.sigma[i] == 0:
                x_norm[i] = (x_norm[i] - self.mu[i]) 
            else:
                x_norm[i] = (x_norm[i] - self.mu[i]) / self.sigma[i]
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

    def getShape(self):
        return self.X.shape, self.y.shape

    def getBiasedX(self):
        return np.hstack((np.ones((len(self.X), 1)), self.X))

    def clone(self, X, y):
        return  Dataset(X, y, self.xlabel, self.ylabel, self.legend, self.title)