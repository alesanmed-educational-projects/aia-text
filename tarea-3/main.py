# -*- coding: utf-8 -*-
import vectorizer
import time
import numpy as np
import pickle
import os

from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from english_stemmer import EnglishTokenizer
from nltk.cluster.util import cosine_distance

categories = ['talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

newsgroups_train = fetch_20newsgroups(subset='all', categories=categories).data

newsgroups_train = newsgroups_train[10:]
    
if not os.path.exists("vectorizer.pickle"):    
    t = time.time()
    text_vectorizer, text_vector = vectorizer.tf_idf_vectorize(newsgroups_train, stop_words='english', tokenizer=EnglishTokenizer())
    print("Vectorizador: {0} s".format(time.time() - t))
    
    pickle.dump(text_vectorizer, open("vectorizer.pickle", "wb"))
    pickle.dump(text_vector, open("vector.pickle", "wb"))
else:
    text_vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    text_vector = pickle.load(open("vector.pickle", "rb"))
    
if not os.path.exists("clusters.pickle"):
    clusters = KMeans()
    t = time.time()
    clusters.fit(text_vector)
    print("K-Medias: {0} s".format(time.time()- t))
    
    pickle.dump(clusters, open("clusters.pickle", "wb"))
else:
    clusters = pickle.load(open("clusters.pickle", "rb"))

while True:
    query = input("Introduzca la consulta:")
    query_vector = text_vectorizer.transform([query])
    
    query_cluster = clusters.predict(query_vector)
    
    documents = text_vector[np.where(query_cluster == clusters.labels_)[0], :]
    
    distances = np.array([[j, cosine_distance(documents[j].toarray()[0], query_vector.toarray()[0])] for j in range(documents.shape[0])])
    distances = distances[distances[:,-1].argsort()][:5]
    
    print("Search results:")
    
    for i in distances:
        print(newsgroups_train[int(i[0])])
        print()
        print("------------------------------------------------")
        print()
