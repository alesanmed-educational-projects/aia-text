# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups


categories = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "sci.space"]

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_all = fetch_20newsgroups(subset='all', categories=categories)

with open("train.txt", "a") as all_file:
     for i in range(len(newsgroups_train.data)):
             all_file.write(newsgroups_train.data[i])
