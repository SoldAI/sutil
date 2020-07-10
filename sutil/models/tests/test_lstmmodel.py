# -*- coding: utf-8 -*-
from sutil.text.TextDataset import TextDataset
from sutil.text.GramTokenizer import GramTokenizer
from sutil.text.OneHotVectorizer import OneHotVectorizer
from sutil.text.PreProcessor import PreProcessor
from sutil.models.LSTMModel import LSTMModel
from sklearn import preprocessing

#filename = "./sutil/datasets/tweets.csv"
#filename = "./sutil/datasets/prestadero_02_mod.csv"
filename = "./sutil/datasets/spanish_names_mod.csv"

#Import data and vectorize it using onetot encoding technique
t_onehot = TextDataset.setvectorizer(filename, OneHotVectorizer({}, GramTokenizer()), PreProcessor.standard(), True)
lb = preprocessing.LabelBinarizer()
lb.fit(t_onehot.y)
train, test = t_onehot.split(0.8, 0)
lstm = LSTMModel(t_onehot.n + 1, 'Simple LSTM example', 20000, 128, len(t_onehot.labels), lb)
lstm.trainModel(train)
lstm.plotHistory()
#Score the model
test_y = lstm.binarizer.transform(test.y)
lstm.score(test.X, test_y)
print(lstm.evaluation)
