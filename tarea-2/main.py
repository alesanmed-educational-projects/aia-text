# -*- coding: utf-8 -*-
import numpy as np
import vectorizer
import stopwords
from spanish_stemmer import SpanishTokenizer

from nltk.cluster.util import cosine_distance

def run(sub_task=1):
    documents = [
        "Éste texto no tiene nada que ver con los demás",
        "La plata fue entregada en camiones color plata",
        "El cargamento de oro llegó en un camión. El cargamento de oro llegó en un camión. El cargamento de oro llegó en un camión",
        "Cargamentos de oro dañados por el fuego",
        "El cargamento de oro llegó en un camión"
    ]
    
    query = ["oro plata camión"]
    if sub_task <= 2:
        text_vectorizer, text_vector = vectorizer.vectorize(documents)
        query_vector = text_vectorizer.transform(query)
        
        if sub_task == 1:
            distances = np.array([np.linalg.norm(text_vector[i].toarray() - query_vector.toarray()) for i in range(text_vector.shape[0])])
        elif sub_task == 2:
            distances = np.array([cosine_distance(text_vector[i].toarray()[0], query_vector.toarray()[0]) for i in range(text_vector.shape[0])])
    elif sub_task >= 3:
        if sub_task == 3:
            text_vectorizer, text_vector = vectorizer.vectorize(documents, stop_words=stopwords.spanish)
        elif sub_task == 4:
            text_vectorizer, text_vector = vectorizer.vectorize(documents, stop_words=stopwords.spanish, tokenizer=SpanishTokenizer())
        elif sub_task == 5:
            text_vectorizer, text_vector = vectorizer.tf_idf_vectorize(documents, stop_words=stopwords.spanish, tokenizer=SpanishTokenizer())
            
        query_vector = text_vectorizer.transform(query)
        
        distances = np.array([cosine_distance(text_vector[i].toarray()[0], query_vector.toarray()[0]) for i in range(text_vector.shape[0])])
    
    min_distance = np.argmin(distances)
    
    print("Documento mas parecido: {0}.\nDistancia: {1}\nTexto del documento:\n{2}".format(min_distance, np.amin(distances), documents[min_distance]))

if __name__ == "__main__":
    for i in range(5):
        run(i+1)