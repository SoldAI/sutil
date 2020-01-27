from sutil.text.GramTokenizer import GramTokenizer
from sutil.text.PhraseTokenizer import PhraseTokenizer

string = "Hi I'm a really helpful string"
t = PhraseTokenizer()
print(t.tokenize(string))

t2 = GramTokenizer()
print(t2.tokenize(string))