from sutil.text.TextVectorizer import TextVectorizer
from sutil.text.PhraseTokenizer import PhraseTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class TFIDFVectorizer(TextVectorizer):

    @classmethod
    def standard(cls):
        return cls({}, PhraseTokenizer())

    def __init__(self, dictionary, tokenizer):
        self.dictionary = dictionary if dictionary else {}
        self.tokenizer = tokenizer
        self.entries = len(dictionary)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.analyzer = tokenizer.tokenize

    def initialize(self, texts):
        self.index = {}
        self.dictionary = self.colapseTFIDF(texts)
        i = 0
        for fn in self.vectorizer.get_feature_names():
                self.index[fn] = i
                i += 1
        self.entries = len(self.dictionary)

    def colapseTFIDF(self, texts):
        vectors = self.vectorizer.fit_transform(texts)
        dense = vectors.todense()
        denselist = np.array(dense.tolist())
        feature_names = self.vectorizer.get_feature_names()
        averaged = np.zeros(len(denselist[0]))
        for i in range(len(denselist[0])):
            averaged[i] = np.average(denselist[:, i])
        #self.dictionary = pd.DataFrame(denselist, columns=feature_names)
        dictionary = {k: v for k, v in zip(feature_names, averaged)}
        return dictionary

    def getValues(self):
        return self.dictionary.values