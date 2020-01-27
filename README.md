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

## PhraseTokenizer
PhraseTokenizer lets you split the tokens of a phrase given a delimiter char. There's also the **GramTokenizer** class whihc lets you split words by fixed amounts of characers.
```python
from sutil.text.GramTokenizer import GramTokenizer
from sutil.text.PhraseTokenizer import PhraseTokenizer

string = "Hi I'm a really helpful string"
t = PhraseTokenizer()
print(t.tokenize(string))

t2 = GramTokenizer()
print(t2.tokenize(string))
```

## TextVectorizer
TextVectorizer class is an abstraction of methods to vectorize text and transform vectors to texts. Sutil implements, **OneHotVectorizer**, **TFIDFVectorizer**, **CountVectorizer**. 
```python
import pandas as pd
from sutil.text.TFIDFVectorizer import TFIDFVectorizer
from sutil.text.GramTokenizer import GramTokenizer
from sutil.text.PreProcessor import PreProcessor
from nltk.tokenize import TweetTokenizer

# Load the data
dir_data = "./sutil/datasets/"
df = pd.read_csv(dir_data + 'tweets.csv')

# Clean the data
patterns = [("\d+", "NUMBER")]
c = [("case", "lower"), ("denoise", "english"), ("stopwords", "english"), ("normalize", patterns)]
p2 = PreProcessor(c)
df['clean_tweet'] = df.tweet.apply(p2.preProcess)

vectorizer = TFIDFVectorizer({}, TweetTokenizer())
vectorizer.initialize(df.clean_tweet)
print(vectorizer.dictionary.head())


vectorizer2 = TFIDFVectorizer({}, GramTokenizer())
vectorizer2.initialize(df.clean_tweet)
print(vectorizer2.dictionary.head())

vector = vectorizer.encodePhrase(df.clean_tweet[0])
print(vectorizer.getValues()[0])
print(vector)

vector2 = vectorizer2.encodePhrase(df.clean_tweet[0])
print(vectorizer2.getValues()[0])
print(vector2)

print(df.clean_tweet[0])
print(vectorizer.decodeVector(vector))
print("*"*50)
print(vectorizer2.decodeVector(vector2))
```
## TextDataSet
**TextDataSet** class abstracts a DataSet made by texts, with includes a vectorizer and a pre processor to pre process the text and transform it from text to vector and from vector to text:

```python
from sutil.text.TextDataset import TextDataset
from sutil.text.GramTokenizer import GramTokenizer
from sutil.text.TFIDFVectorizer import TFIDFVectorizer
from sutil.text.PhraseTokenizer import PhraseTokenizer
from sutil.text.PreProcessor import PreProcessor

# Load the data in the standard way
filename = "./sutil/datasets/tweets.csv"
t = TextDataset.standard(filename, ",")
print(t.texts)
print(t.X)
print(t.shape)
print(t.X[0])
print(t.vectorizer.index)
print(t.vectorizer.encodePhrase("united oh"))

x = input("Presione enter para continuar...")
# Creaate the dataset witha custom vectorizer and pre processor
patterns = [("\d+", "NUMBER")]
c = [("case", "lower"), ("denoise", "spanish"), ("stopwords", "spanish"), ("normalize", patterns)]
preprocessor = PreProcessor(c)
vectorizer = TFIDFVectorizer({}, GramTokenizer())
t2 = TextDataset.setvectorizer(filename, vectorizer, preprocessor)
print(t2.texts)
print(t2.X)
print(t2.shape)
print(t2.X[0])
vector = t2.encodePhrase("United oh the")
i = 0
for v in vector:
	if v != 0:
		print(v)
		print(i)
	i += 1
print(t2.vectorizer.decodeVector(vector))

x = input("Presione enter para continuar...")
patterns = [("\d+", "NUMBER")]
c = [("case", "lower"), ("denoise", "spanish"), ("stopwords", "english"), ("normalize", patterns)]
pre2 = PreProcessor(c)
vectorizer = TFIDFVectorizer({}, PhraseTokenizer())
t3 = TextDataset.setvectorizer(filename, vectorizer, pre2)
print(t3.texts)
print(t3.X)
print(t3.shape)
print(t3.X[0])
vector = t3.encodePhrase("United oh the")
i = 0
for v in vector:
	if v != 0:
		print(v)
		print(i)
	i += 1
print(t3.decodeVector(vector))
```
