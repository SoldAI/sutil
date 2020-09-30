#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 00:02:48 2020

@author: mcampos
"""

import pandas as pd
from sutil.text.TextVectorizer import TextVectorizer
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

vectorizer = TextVectorizer({}, TweetTokenizer())
vectorizer.initialize(df.clean_tweet)
print(vectorizer.dictionary.keys())
print(df.clean_tweet[0])
vector = vectorizer.encodePhrase(df.clean_tweet[0])
print(vector)
print(vectorizer.decodeVector(vector))
print("*"*50)