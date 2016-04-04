# -*- coding: utf-8 -*-
from nltk import word_tokenize          
from nltk.stem import SnowballStemmer 

class SpanishTokenizer(object):
    def __init__(self):
        self.sbs = SnowballStemmer(language="spanish")
    def __call__(self, doc):
        return [self.sbs.stem(t) for t in word_tokenize(doc)]