# -*- coding: utf-8 -*-
import numpy as np
from sutil.base.Dataset import Dataset
from sklearn.linear_model import LogisticRegression
from sutil.base.Experiment import Experiment
from sutil.models.SklearnModel import SklearnModel
from sutil.models.RegularizedLogisticRegression import RegularizedLogisticRegression
from sutil.neuralnet.NeuralNetworkClassifier import NeuralNetworkClassifier

# Load the data
datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
d.normalizeFeatures()
print("Size of the dataset... ")
print(d.size)
sample = d.sample(0.3)
print("Size of the sample... ")
print(d.sample)


# Create the models
theta = np.zeros((d.n + 1, 1))
lr = RegularizedLogisticRegression(theta, 0.03, 0)
m = SklearnModel('Sklearn Logistic', LogisticRegression())
# Look for the best parameters using a sample
nn = NeuralNetworkClassifier((d.n, len(d.labels)))
nn.searchParameters(sample)

input("Press enter to continue...")

# Create the experiment
experiment = Experiment(d, None, 0.8, 0.2)
experiment.addModel(lr, name = 'Sutil Logistic Regression')
experiment.addModel(m, name = 'Sklearn Logistic Regression')
experiment.addModel(nn, name = 'Sutil Neural Network')

# Run the experiment
experiment.run(plot = True)