import numpy as np

class TextVectorizer():

    def __init__(self, dictionary, tokenizer):
        self.dictionary = dictionary if dictionary else {}
        self.tokenizer = tokenizer
        self.entries = len(dictionary)
        self.index = {}
        for k, v in self.dictionary.items():
            if not k in self.index.keys():
                self.index[k] = len(self.index)

    def initialize(self, texts):
        self.index = {}
        for t in texts:
            for token in self.tokenizer.tokenize(t):
                if token in self.dictionary.keys():
                    self.dictionary[token] += 1 
                else:
                    self.dictionary[token] = 1
                    self.index[token] = len(self.index)
        # Encode the reciprocal
        for k in self.dictionary.keys():
            self.dictionary[k] = 1 / (self.dictionary[k] + 1)
        self.entries = len(self.dictionary)

    def encodePhrase(self, phrase):
        vector = np.zeros(len(self.dictionary))
        for t in self.tokenizer.tokenize(phrase):
            if t in self.dictionary.keys():
                vector[self.index[t]] = self.dictionary[t]
        return vector

    def decodeVector(self, vector):
        inv_dict = {v: k for k, v in self.index.items()}
        out = ""
        for i in range(len(vector)):
            if vector[i] != 0:
                out += str(inv_dict[i]) + ","
        return out[: -1]

    def textToMatrix(self, texts):
        print("Texts: " + str(len(texts)))
        print("Dictionary: " + str(len(self.dictionary)))
        matrix = np.zeros((len(texts), self.entries))
        if len(self.dictionary) < 1:
            return np.array([0])
        i = 0
        for t in texts:
            matrix[i:] = self.encodePhrase(t)
            i += 1
            if i % 500 == 0:
                print(i)
        return matrix