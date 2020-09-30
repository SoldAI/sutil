# -*- coding: utf-8 -*-
import numpy as np
from sutil.base.Dataset import Dataset
from sutil.models.RegularizedLogisticRegression import RegularizedLogisticRegression
from sutil.metrics.ModelROC import ModelROC

datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
theta = np.zeros((d.n + 1, 1))
alpha = 0.03
l = 0
lr = RegularizedLogisticRegression(theta, alpha, l)
lr.trainModel(d)

m = ModelROC(lr, d.getBiasedX(), d.y, legend = 'Example of Model ROC usage')
m.plot()
m.zoom((0, 0.4),(0.5, 1.0))