from sutil.text.PhraseTokenizer import PhraseTokenizer

class GramTokenizer(PhraseTokenizer):

    def __init__(self, gram_length=3, space_char='_'):
        self.gram_length = gram_length
        self.space_char = space_char

    def getTokens(self, text, delimiter = " "):
        """
        This method extract the grams of the string
        """
        #copy = super(GramTokenizer, self).clean_string(text)
        copy = text.replace(" ", self.space_char)

        if len(copy) <= self.gram_length:
            grams = [copy.ljust(self.gram_length, self.space_char)]
        else:
            grams_number = len(copy) - self.gram_length + 1
            grams = [copy[i:i+self.gram_length] for i in range(grams_number)]
        return grams