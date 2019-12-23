# -*- coding: utf-8 -*-
import numpy as np
from sutil.base.Dataset import Dataset
from sutil.models.SklearnModel import SklearnModel
from sutil.models.RegularizedLogisticRegression import RegularizedLogisticRegression
from sklearn.linear_model import LogisticRegression
from sutil.neuralnet.NeuralNetworkClassifier import NeuralNetworkClassifier

datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
theta = np.zeros((d.n + 1, 1))
alpha = 0.03
l = 0
lr = RegularizedLogisticRegression(theta, alpha, l)
lr.trainModel(d)
lr.score(d.X, d.y)
lr.roc.plot()
lr.roc.zoom((0, 0.4),(0.5, 1.0))

input("Press enter to continue...")

ms = LogisticRegression()
m = SklearnModel('Sklearn Logistic', ms)
m.trainModel(d)
m.score(d.X, d.y)
m.roc.plot()
m.roc.zoom((0, 0.4),(0.5, 1.0))

input("Press enter to continue...")
d.normalizeFeatures()
print(d.size)
input("Size of the dataset... ")
sample = d.sample(0.3)
print(sample.size)
input("Size of the sample... ")
sample2 = d.sample(examples = 30)
print(sample2.size)
input("Size of the sample 2... ")
nn = NeuralNetworkClassifier((d.n, len(d.labels)))
nn.searchParameters(sample2)
nn.trainModel(d)
nn.score(d.X, d.y)
nn.roc.plot()
nn.roc.zoom((0, 0.4),(0.5, 1.0))
