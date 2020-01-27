# -*- coding: utf-8 -*-
from sutil.base.Dataset import Dataset
from sutil.text.PreProcessor import PreProcessor
from sutil.text.PhraseTokenizer import PhraseTokenizer
from sutil.text.OneHotVectorizer import OneHotVectorizer
import pandas as pd

class TextDataset(Dataset):

    @classmethod
    def standard(cls, filename, delimiter):
        preprocessor = PreProcessor.standard()
        tokenizer = PhraseTokenizer()
        vectorizer = OneHotVectorizer({}, tokenizer)
        df = pd.read_csv(filename)
        X = df[df.columns[0]]
        y = df[df.columns[-1]].values
        return cls(X, y, vectorizer, preprocessor)

    @classmethod
    def setvectorizer(cls, filename, vectorizer = None, preprocessor = None, **kwargs):
        df = pd.read_csv(filename)
        X = df[df.columns[0]]
        y = df[df.columns[-1]].values
        return cls(X, y, vectorizer, preprocessor)

    def __init__(self, X, y, vectorizer = None, preprocessor = None, **kwargs):
        if vectorizer:
            self.vectorizer = vectorizer 
        else: 
            self.vectorizer = OneHotVectorizer()
        if preprocessor:
            self.preprocessor = preprocessor 
        else: 
            self.preprocessor = PreProcessor.standard()
        print(X)
        self.initialize(X, y)

    def initialize(self, texts, y, **kwargs):
        self.texts = texts
        processed = self.preprocessor.batchPreProcess(texts)
        self.vectorizer.initialize(processed)
        X = self.vectorizer.textToMatrix(texts)
        print(X)
        print(y)
        self.setData(X, y)
        self.X, self.mu, self.sigma = self.normalizeFeatures()

    def encodePhrase(self, phrase):
        phrase = self.preprocessor.preProcess(phrase)
        return self.vectorizer.encodePhrase(phrase)

    def decodeVector(self, vector):
        return self.vectorizer.decodeVector(vector)
