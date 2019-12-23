# -*- coding: utf-8 -*-
import numpy as np
from sutil.base.Dataset import Dataset
from sutil.models.SklearnModel import SklearnModel
from sutil.models.RegularizedLogisticRegression import RegularizedLogisticRegression
from sklearn.linear_model import LogisticRegression

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

def simple():
    datafile = './sutil/datasets/ex2data1.txt'
    d = Dataset.fromDataFile(datafile, ',')
    theta = np.zeros((d.n + 1, 1))
    lr = RegularizedLogisticRegression(theta, 0.03, 0)
    lr.trainModel(d)
    lr.score(d.X, d.y)

def sk():
    datafile = './sutil/datasets/ex2data1.txt'
    d = Dataset.fromDataFile(datafile, ',')
    ms = LogisticRegression()
    m = SklearnModel('Sklearn Logistic', ms)
    m.trainModel(d)
    m.score(d.X, d.y)

import timeit
print(timeit.timeit(simple, number=100))
input("Press enter to continue...")
print(timeit.timeit(sk, number=100))