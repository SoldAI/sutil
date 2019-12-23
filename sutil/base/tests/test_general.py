######### Dataset test##########
from sutil.base.Dataset import Dataset

datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
print(d.size)

sample = d.sample(0.3)
print(sample.size)
        
sample.save("modelo_01")

train, validation, test = d.split(train = 0.8, validation = 0.2)
print(train.size)
print(validation.size)
print(test.size)

##########Regularized Logistic Regression ############
import numpy as np
from sutil.base.Dataset import Dataset
from sutil.models.RegularizedLogisticRegression import RegularizedLogisticRegression

datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
d.xlabel = 'Exam 1 score'
d.ylabel = 'Exam 2 score'
d.legend = ['Admitted', 'Not admitted']
iterations = 400
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
d.plotData()

theta = np.zeros((d.n + 1, 1))
lr = RegularizedLogisticRegression(theta, 0.03, 0, train=1)
lr.trainModel(d)
lr.score(d.X, d.y)
lr.roc.plot()
lr.roc.zoom((0, 0.4),(0.5, 1.0))

###########Sklearn Model###############
import numpy as np
from sutil.base.Dataset import Dataset
from sutil.models.SklearnModel import SklearnModel
from sklearn.linear_model import LogisticRegression

datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
ms = LogisticRegression()
m = SklearnModel('Sklearn Logistic', ms)
m.trainModel(d)
m.score(d.X, d.y)
m.roc.plot()
m.roc.zoom((0, 0.4),(0.5, 1.0))

#########Neural Network#############
from sutil.base.Dataset import Dataset
from sutil.neuralnet.NeuralNetworkClassifier import NeuralNetworkClassifier

datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
d.normalizeFeatures()
sample2 = d.sample(examples = 30)

nn = NeuralNetworkClassifier((d.n, len(d.labels)))
nn.searchParameters(sample2)
nn.trainModel(d)
nn.score(d.X, d.y)
nn.roc.plot()
nn.roc.zoom((0, 0.4),(0.5, 1.0))

##########Experiment#############
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
