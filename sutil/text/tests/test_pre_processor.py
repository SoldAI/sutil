# -*- coding: utf-8 -*-
from sutil.text.PreProcessor import PreProcessor

def testPreProcessor(string, configuration = None):
	p = PreProcessor.standard()
	print("*"*40)
	if configuration:
		p = PreProcessor(configuration)
	print(string)
	print(p.preProcess(string))
	print("*"*40)

string = "La Gata maullaba en la noche $'@|··~½¬½¬{{[[]}aqAs   qasdas 1552638"

testPreProcessor(string)

patterns = [("\d+", "NUMBER")]
c = [("case", "lower"), ("denoise", "spanish"), ("stopwords", "spanish"), 
     ("stem", "spanish"), ("lemmatize", "spanish"), ("normalize", patterns)]
testPreProcessor(string, c)

c = [("case", "lower"), ("denoise", "spanish"), ("stem", "spanish"), 
     ("normalize", patterns), ("callable", lambda x: x.replace(" ", "_"))]
testPreProcessor(string, c)