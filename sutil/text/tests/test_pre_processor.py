# -*- coding: utf-8 -*-
import numpy as np
from sutil.text.PreProcessor import PreProcessor

string = "La Gata maullaba en la noche $'@|··~½¬½¬{{[[]}aqAs   qasdas 1552638"
p = PreProcessor.standard()
print(p.preProcess(string))

patterns = [("\d+", "NUMBER")]
c = [("case", "lower"), ("denoise", "spanish"), ("stopwords", "spanish"), ("stem", "spanish"), ("lemmatize", "spanish"), ("normalize", patterns)]
p2 = PreProcessor(c)
print(p2.preProcess(string))


c = [("case", "lower"), ("denoise", "spanish"), ("stem", "spanish"), ("normalize", patterns)]
p3 = PreProcessor(c)
print(p3.preProcess(string))