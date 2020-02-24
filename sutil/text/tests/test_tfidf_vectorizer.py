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