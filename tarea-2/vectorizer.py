# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize(documents, stop_words = None, tokenizer = None):
    vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=tokenizer)
    vector = vectorizer.fit_transform(documents)
    
    return vectorizer, vector
    
def tf_idf_vectorize(documents, stop_words = None, tokenizer = None):
    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenizer)
    vector = vectorizer.fit_transform(documents)
    
    return vectorizer, vector