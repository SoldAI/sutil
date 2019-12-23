import numpy as np
#import soldaimltk.general.Tokenizer as Tokenizer

class OneHotEncoder:

    def __init__(self, dictionary, tokenizer):
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.entries = len(dictionary)

    def encodePhrase(self, phrase):
        vector = np.zeros(len(self.dictionary))
        for t in self.tokenizer.getTokens(phrase):
            if t in self.dictionary.keys():
                vector[self.dictionary[t]] = 1
        return vector
    
    def encodePhrase2(self, phrase):
        vector = np.zeros(len(self.dictionary))
        for t in self.tokenizer.getTokens(phrase):
            if t in self.dictionary.keys():
                vector[self.dictionary[t]] += 1
        return vector
    
    def decodeVector(self, vector):
        inv_dict = {v: k for k, v in self.dictionary.items()}
        out = ""
        for i in len(vector):
            if vector[i] == 1:
                out += str(inv_dict[i]) + ","
        return out

    def addTermToDictionary(self, term):
        if not term in self.dictionary:
            self.dictionary[term] = len(self.dictionary)
            self.entries = len(self.dictionary)