import string
import unicodedata

class PhraseTokenizer():

    def __init__(self, delimiter = None):
        self.delimiter = delimiter if delimiter else " "

    def getTokens(self, text, delimiter = " "):
        return text.split(delimiter)

    def tokenize(self, text):
        return self.getTokens(text)
