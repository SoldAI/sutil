import numpy as np

class TextVectorizer():

    def __init__(self, dictionary, tokenizer):
        self.dictionary = dictionary if dictionary else {}
        self.tokenizer = tokenizer
        self.entries = len(dictionary)

    def initialize(self, texts):
        for t in texts:
            tokens = self.tokenizer(t)
            for token in tokens:
                if token in self.dictionary.keys():
                    self.dictionary[token] += 1 
                else:
                    self.dictionary[token] = 1

    def encodePhrase(self, phrase):
        vector = np.zeros(len(self.dictionary))
        for t in self.tokenizer.getTokens(phrase):
            if t in self.dictionary.keys():
                vector[self.dictionary[t]] = 1
        return vector

    def decodeVector(self, vector):
        inv_dict = {v: k for k, v in self.dictionary.items()}
        out = ""
        for i in len(vector):
            if vector[i] == 1:
                out += str(inv_dict[i]) + ","
        return out

    def textToMatrix(self, texts):
        if self.entries < 1:
            return np.array([0])

        matrix = np.zeros((len(texts), self.entries))
        i = 0
        for t in texts:
            matrix[i:] = self.encodePhrase(t)
            i += 1
        return matrix