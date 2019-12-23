import string
import unicodedata

class PhraseTokenizer():

    def __init__(self, space_char='_', additional_chars='ñ'):
        self.space_char = space_char
        self.additional_chars = additional_chars
        self.valid_chars = string.ascii_lowercase + string.digits + \
                            self.space_char + self.additional_chars

    def cleanString(self, text):
        sanitized = text.lower()
        sanitized = sanitized.replace(self.space_char,' ')
        sanitized = self.space_char.join(sanitized.split())
        chars = [c if c in self.valid_chars else unicodedata.normalize('NFD', c)[0] for c in sanitized]
        sanitized = ''.join([c for c in chars if c in self.valid_chars])
        return sanitized 


    def getTokens(self, text, delimiter = " "):
        return self.cleanString(text).split(delimiter)
