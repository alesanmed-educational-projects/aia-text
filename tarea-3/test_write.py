# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups

categories = ['talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

newsgroups_all = fetch_20newsgroups(subset='all', categories=categories)

with open("all.txt", "a") as all_file:
     for i in range(len(newsgroups_all.data)):
             all_file.write(newsgroups_all.data[i])
