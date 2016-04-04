# -*- coding: utf-8 -*-
import numpy as np
import itertools
import imageio
import os
import time

from scipy import cluster
from sklearn.datasets import load_iris
import kmeans
from matplotlib import pyplot as plt
from os import mkdir

iris = load_iris()
data = iris.data
names = iris.feature_names

np.random.shuffle(data)
k = 3

t = time.time()
centers, center_steps = kmeans.k_means(data, k=k, distance='e')
print(time.time() - t)

for step in range(len(center_steps)):
    print("{0}/{1}".format(step + 1, len(center_steps)))
    for combination in itertools.combinations(range(data.shape[1]), 2):
        step_centers = np.array(center_steps[step])
        assignment, cdist = cluster.vq.vq(data[:, combination], 
                                          step_centers[:, combination])

        if not os.path.exists(names[combination[0]] + "-" 
            + names[combination[1]]):
            mkdir(names[combination[0]] + "-" + names[combination[1]])
        
        for i in range(k):
            data = np.append(data, [step_centers[i]], axis=0)
            assignment = np.append(assignment, -1)
        
        fig = plt.figure()
        plt.scatter(data[:, combination[0]], 
                    data[:, combination[1]], c=assignment)
        fig.suptitle('Paso {0}'.format(step + 1))
        plt.xlabel('{0}'.format(names[combination[0]]))
        plt.ylabel('{0}'.format(names[combination[1]]))
        plt.savefig('{0}/step_{1}.png'.format(
                                        names[combination[0]] + "-" 
                                        + names[combination[1]], step), 
                    dpi=271, bbox_inches='tight')
        fig.clear()
        plt.close('all')

i = 1
for subdir, dirs, files in os.walk('.'):
    if subdir == '.' or "__" in subdir:
        continue
    
    print("Gif {0}".format(i))
    i += 1
    print(subdir)
    
    with imageio.get_writer(
        os.path.join(subdir, 
                     'steps.gif'), 
        mode='I') as writer:    
            for filename in sorted(files):
                if ".png" in filename:
                    print(filename)
                    for _ in range(5):
                        image = imageio.imread(os.path.join(subdir, filename))
                        writer.append_data(image)
