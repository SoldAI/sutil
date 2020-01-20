#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday Jan 17 18:28:06 2020
pre processor class to clean text
@author: mcampos
"""
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sutil.text.StringJanitor import StringJanitor
from nltk.corpus import stopwords
import re

class PreProcessor:

    @classmethod
    def standard(cls):
        configuration = [("case", "lower"),
                         ("denoise", "spanish")]
        return cls(configuration)

    def __init__(self, configurations):
        self.actions = []
        m2m = {"case": "caseNormalization", 
                       "denoise": "removeNoise", 
                       "stopwords": "stopWordsRemoval",
                       "stem": "stem",
                       "lemmatize": "lemmatize",
                       "normalize":"normalize"}
        for entry in configurations:
        	self.actions.append((m2m[entry[0]], entry[1]))
        self.lemmatizer = WordNetLemmatizer()
        self.janitor = StringJanitor.spanish()
        #self.janitor.space_char = " "
        self.stemmer = PorterStemmer()

    def preProcess(self, string):
        result = string
        for a in self.actions:
            print("Performing " + a[0])
            method = getattr(self, a[0])
            result = method(a[1], result)
            print(result)
        return result

    def stopWordsRemoval(self, idiom, string):
        sw = set(stopwords.words(idiom))
        words = string.split(" ")
        cleaned_words = [w for w in words if w not in sw]
        return " ".join(cleaned_words)

    def removeNoise(self, idiom, string):
        cleaned_string = self.janitor.clean(string)
        return cleaned_string.replace("_", " ")

    def caseNormalization(self, type, string):
        if type == "lower":
            return string.lower()

    def stem(self, idiom, string):
        raw_words = string.split(" ")
        stemmed_words = [self.stemmer.stem(word=word) for word in raw_words]
        return " ".join(stemmed_words)

    def lemmatize(self, idiom, string):
        words = string.split(" ")
        lemmatized_words = [self.lemmatizer.lemmatize(word=word, pos='v') for word in words]
        return " ".join(lemmatized_words)

    #Simple text normalization using regular expressions
    def normalize(self, patterns, string):
        normalized = string
        for p in patterns:
            normalized = re.sub(p[0], p[1], normalized)
        return normalized