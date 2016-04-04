# -*- coding: utf-8 -*-
import numpy as np
from scipy import cluster
from matplotlib import pyplot

np.random.seed(123)
tests = np.reshape( np.random.uniform(0,100,60), (30,2) )


cent, var = cluster.vq.kmeans(tests,3)

#use vq() to get as assignment for each obs.
assignment, cdist = cluster.vq.vq(tests,cent)
print(assignment)

pyplot.scatter(tests[:,0], tests[:,1], c=assignment)
pyplot.show()
