# -*- coding: utf-8 -*-
from nltk import word_tokenize          
from nltk.stem import SnowballStemmer 

class EnglishTokenizer(object):
    def __init__(self):
        self.sbs = SnowballStemmer(language="english")
    def __call__(self, doc):
        return [self.sbs.stem(t) for t in word_tokenize(doc)]