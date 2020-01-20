# sutil
This repository contains a set of tools to deal with machine learning and natural language processing tasks, including classes to make quick experimentation of different classifacation models.

## Dataset
This class is made to load csv styles dataset's where all the features are comma separeted and the class is in the last column.
It includes functions to normalize the features, add bias, save the data to a file and load from it. Also includes functions
to split the train, validation and test datasets.

```python
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
```

## Regularized Logistic Regression
You can also include your own models as a Regularized Logistic Regression, implemented manually using numpy and included in the sutil.models package
```python
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
```

## Sklearn model 
You can also embed the sklearn models in a wrapper class in order to run experiments with diferent models implemented in sklearn. In the same style you can create tensorflow, keras or pytorch models inhyereting from sutil.modes.Model class and
implementing the trainModel and predict methods.
```python
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
```

## Neural Network Classifer
This class let's you perform classifcation using a Neural Network, multiperceptron classifer. It wraps the sklearn MLPClassifer 
and implements a method to search different activations, solvers and hidden layers structures. Upu can pass 
your own arguments to initialize the network as you want.
```python
from sutil.base.Dataset import Dataset
from sutil.neuralnet.NeuralNetworkClassifier import NeuralNetworkClassifier

datafile = './sutil/datasets/ex2data1.txt'
d = Dataset.fromDataFile(datafile, ',')
d.normalizeFeatures()
sample = d.sample(examples = 30)

nn = NeuralNetworkClassifier((d.n, len(d.labels)))
nn.searchParameters(sample)
nn.trainModel(d)
nn.score(d.X, d.y)
nn.roc.plot()
```

## Experiment
The experiment class let's you perform the data split and test against different models to compare the 
performance automatically
```python
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
```

# Text utilities
Sutil includes text utilities to process and transform text for classification

## PreProcessor
Pre processor class let's you implement text pre processinf functions to transform the data. It wraps nltk methods and uses it's own methods toperform:
* Case normalization
* Noise removal
* Stemming
* Leammatizing
* Pattern text normalization
```python
from sutil.text.PreProcessor import PreProcessor

string = "La Gata maullaba en la noche $'@|··~½¬½¬{{[[]}aqAs   qasdas 1552638"
p = PreProcessor.standard()
print(p.preProcess(string))

patterns = [("\d+", "NUMBER")]
c = [("case", "lower"), ("denoise", "spanish"), ("stopwords", "spanish"), 
     ("stem", "spanish"), ("lemmatize", "spanish"), ("normalize", patterns)]
p2 = PreProcessor(c)
print(p2.preProcess(string))


c = [("case", "lower"), ("denoise", "spanish"), ("stem", "spanish"), 
     ("normalize", patterns)]
p3 = PreProcessor(c)
print(p3.preProcess(string))
```
