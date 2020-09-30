import numpy as np
from sutil.text.TextVectorizer import TextVectorizer

class OneHotVectorizer(TextVectorizer):

    def initialize(self, texts):
        self.index = {}
        for t in texts:
            tokens = self.tokenizer.tokenize(t)
            for token in tokens:
                self.addTermToDictionary(token)

    def addTermToDictionary(self, term):
        if not term in self.dictionary:
            self.dictionary[term] = 1
            self.entries += 1
            self.index[term] = len(self.index)