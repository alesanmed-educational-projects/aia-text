# -*- coding: utf-8 -*-
import vectorizer
import stopwords
import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from spanish_stemmer import SpanishTokenizer
from nltk.cluster.util import cosine_distance

categories = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "sci.space"]

query = ["From: revpk@cellar.org (Brian \'Rev P-K\' Siano)\nSubject: Novell and Windows 3.1\nOrganization: The Cellar BBS and public access system\nLines: 35\n\n\tI\'m working at a workstation which is usually attached to a Novell\nnetwork (using shell version 3.22, I think). The workstation, a 386, was set\nup to run Windows 3.0 with the network about a year ago. Needless to say,\nI\'d like to upgrade it to Windows 3.1, and have it work with the network.\n\n\tBasically, the Windows files\'d be on the local hard drive, but\nseveral DOS applications, like Word Perfect, will be on the network. I\'d\nmainly want Windows to access the network drives, the network printers, and\nperhaps handle some network functions as well. If I could multitask the DOS\napps whose executables are on the network, that\'d be great, but I could live\nwithout it.\n\n\tEventually, I\'d like to get a few other 486s in the office working\nwith the network and Windows 3.1 as well. (However, most of the terminals\nare 286s, which leaves the network pretty much DOS-bound, and I guess that\nleaves out Windows for Workgroups.) And in the future, maybe there\'d be\nNorton desktop, but that\'s gettingahead of myself.\n\n\tAs you can guess, I\'ve never done anything like this before. I\'ve\nread through the networks material that came with Windows, but still, I\'d\nlike to know if anyone out there has any experience in such an area.\n\n\tPlease reply by Email. I don\'t scan these newsgroups often.\n\n\tThanks for any replies.\n\n\nBrian \"Rev. P-K\" Siano                                  revpk@cellar.org\n\n\"Well, I\'ll know right away by the look in her eyes\nshe\'s lost all illusions and she\'s worldly wise, and I know\nif I give her a listen, she\'s what I\'ve been missing, what I\'ve been missing\nI\'ll be lost in love and havin\' some fun with my cynical girl\nWho\'s got no use for the real world, I\'m looking for a Cynical Girl\"\n          --- Marshall Crenshaw, \"Cynical Girl\"\n"]


newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

t = time.time()
text_vectorizer, text_vector = vectorizer.tf_idf_vectorize(newsgroups_train.data, stop_words=stopwords.spanish, tokenizer=SpanishTokenizer())
query_vector = text_vectorizer.transform(query)
print("Vectorizador: {0} s".format(time.time() - t))

clusters = KMeans()
t = time.time()
clusters.fit(text_vector)
print("K-Medias: {0} s".format(time.time()- t))

query_cluster = clusters.predict(query_vector)

documents = text_vector[np.where(query_cluster == clusters.labels_)[0], :]

distances = np.array([cosine_distance(documents[i].toarray()[0], query_vector.toarray()[0]) for i in range(documents.shape[0])])
    
min_distance = np.argmin(distances)

print("Documento mas parecido: {0}.\nDistancia: {1}\nTexto del documento:\n{2}".format(min_distance, np.amin(distances), newsgroups_train.data[min_distance]))
