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

newsgroups_obj = fetch_20newsgroups(subset='all', categories=categories)

newsgroups = newsgroups_obj.data

newsgroups_train = newsgroups[10:]
    
if not os.path.exists("vectorizer_test.pickle") or not os.path.exists("vector_test.pickle"):
    t = time.time()
    text_vectorizer, text_vector = vectorizer.tf_idf_vectorize(newsgroups_train, stop_words='english', tokenizer=EnglishTokenizer())
    print("Vectorizador: {0} s".format(time.time() - t))
    
    pickle.dump(text_vectorizer, open("vectorizer_test.pickle", "wb"))
    pickle.dump(text_vector, open("vector_test.pickle", "wb"))
else:
    text_vectorizer = pickle.load(open("vectorizer_test.pickle", "rb"))
    text_vector = pickle.load(open("vector_test.pickle", "rb"))
    
if not os.path.exists("clusters_test.pickle"):
    clusters = KMeans()
    t = time.time()
    clusters.fit(text_vector)
    print("K-Medias: {0} s".format(time.time()- t))
    
    pickle.dump(clusters, open("clusters_test.pickle", "wb"))
else:
    clusters = pickle.load(open("clusters_test.pickle", "rb"))

i=1
print("Porcentaje de grupos similares entre los documentos")
for query in newsgroups[:10]:
    query_vector = text_vectorizer.transform([query])
    
    query_cluster = clusters.predict(query_vector)
    
    documents = text_vector[np.where(query_cluster == clusters.labels_)[0], :]
    
    distances = np.array([[j, cosine_distance(documents[j].toarray()[0], query_vector.toarray()[0])] for j in range(documents.shape[0])])
    distances = distances[distances[:,-1].argsort()][:5]

    groups = [newsgroups_obj.target[int(j[0])] for j in distances]
    
    groups, unique_counts = np.unique(groups, return_counts=True)
    
    percentages = [j / np.sum(unique_counts) for j in unique_counts]
    
    print("Documento {0}, Grupo mas frecuente: {1}-{2}%".format(i, groups[np.argmax(percentages)], np.amax(percentages)*100))
    i += 1

i= 1
print("Porcentaje de grupos similares al de la consulta")
for query in newsgroups[:10]:
    query_vector = text_vectorizer.transform([query])
    
    query_cluster = clusters.predict(query_vector)
    
    documents = text_vector[np.where(query_cluster == clusters.labels_)[0], :]
    
    distances = np.array([[i, cosine_distance(documents[i].toarray()[0], query_vector.toarray()[0])] for i in range(documents.shape[0])])
    distances = distances[distances[:,-1].argsort()][:5]

    groups = [newsgroups_obj.target[int(j[0])] for j in distances]
    
    similar_groups = np.where(groups == newsgroups_obj.target[i-1])[0].size
    
    print("Documento {0}, porcentaje de grupos similares al original {1}%".format(i, (similar_groups / distances.size) * 100))
    i += 1
    