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
        vectors = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        self.dictionary = pd.DataFrame(denselist, columns=feature_names)

    def encodePhrase(self, phrase):
         tokens = self.tokenizer.tokenize(phrase)
         vec = np.zeros(self.dictionary.shape[1])
         for t in tokens:
             #print("including token " + t)
             if t in self.dictionary.columns:
                 index = self.dictionary.columns.get_loc(t)
                 #print(index, self.dictionary[t][0])
                 vec[index] = self.dictionary[t][0]
         return vec

    def decodeVector(self, vector):
        out = ""
        for i in range(len(vector)):
            if vector[i] != 0:
                out += self.dictionary.columns[i]
        return out

    def getValues(self):
        return self.dictionary.values