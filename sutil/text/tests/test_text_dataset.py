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
