# -*- coding: utf-8 -*-
from sutil.base.Dataset import Dataset
from sutil.text.PreProcessor import PreProcessor
from sutil.text.PhraseTokenizer import PhraseTokenizer
from sutil.text.OneHotVectorizer import OneHotVectorizer
import pandas as pd

class TextDataset(Dataset):

    @classmethod
    def fromDataFile(cls, filename, delimiter):
        data = np.loadtxt(filename, delimiter=delimiter)
        X = data[:, 0:-1]
        y = data[:, -1]
        return cls(X, y)

    @classmethod
    def standard(cls, filename, delimiter):
        preprocessor = PreProcessor.standard()
        tokenizer = PhraseTokenizer()
        vectorizer = OneHotVectorizer({}, tokenizer)
        df = pd.read_csv(filename)
        X = df.columns[0]
        y = df.columns[-1]
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
        self.initialize(X, y, **kwargs)

    def initialize(self, texts, y, **kwargs):
        self.texts = texts
        processed = self.preprocessor.batchPreProcess(texts)
        self.vectorizer.initialize(processed)
        self.vectorize(texts)
        self.setData(self.X, y, kwargs)

    def vectorize(self, texts):
        self.X = self.vectorizer.textToMatrix(texts)
        self.n = self.X.shape[1]
        self.X, self.mu, self.sigma = self.normalizeFeatures()