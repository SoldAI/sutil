import numpy as np
from sutil.text.TextVectorizer import TextVectorizer

class OneHotVectorizer(TextVectorizer):

    def __init__(self, dictionary, tokenizer):
        self.dictionary = dictionary if dictionary else {}
        self.tokenizer = tokenizer
        self.entries = len(dictionary)

    def initialize(self, texts):
        for t in texts:
            tokens = self.tokenizer.tokenize(t)
            for token in tokens:
                self.addTermToDictionary(token)

    def addTermToDictionary(self, term):
        if not term in self.dictionary:
            self.dictionary[term] = self.entries
            self.entries += 1