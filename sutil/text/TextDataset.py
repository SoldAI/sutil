# -*- coding: utf-8 -*-
import datetime
from sutil.base.Dataset import Dataset
import pandas as pd
import sklearn.feature_extraction.text.TfidfVectorizer
import sklearn.feature_extraction.text.CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

class TextDataset(Dataset):

    def __init__(self, X, y, vectorizer = None, **kwargs):
        self.vectorizer = vectorizer if vectorizer is not None TfidfVectorizer()
        self.initialize(X, y, **kwargs)
    
    def SimpleTokenizer(text):
        #Filtering chars not in A to z maybe use the string janitor
        charfilter = re.compile('[a-zA-Z]+');
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words if word not in stopWords]
        ntokens = list(filter(lambda token: charfilter.match(token),words))
        return ntokens

    def vectorize(self):
        self.rawX = X
        self.X = self.vectorizer.fit_transform(self.X)
        self.X, self.mu, self.sigma = self.normalizeFeatures()
        
    def vectorizeText(self, text):
        words = text.split(" ")
        vec = np.zeros(self.X.shape[1])
        
        for w in words:
            print("including word " + w)
            
            if w in tfidf.columns:
                index = tfidf.columns.get_loc(w)
                print(index, tfidf[w][row])
                vec[index] = tfidf[w][row]
        
        if self.isNormalized():
            vec = self.normalizeExample(vec)
        return vec
    
        