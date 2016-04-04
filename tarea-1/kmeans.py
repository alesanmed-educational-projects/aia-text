# -*- coding: utf-8 -*-
import numpy as np
import random

from scipy.spatial.distance import cityblock, hamming
from numpy.linalg import norm as euclidean


def k_means(data, k=2, distance='e'):
    centers = np.array(random.sample(list(data), k))
    
    centers_steps = [centers.tolist()]
    
    changed = True
    while changed:
        prev_centers = np.copy(centers)
        data_nr = data.shape[0]
        clusters = np.empty((data_nr, k))
        for i in range(data_nr):
            if distance == 'e':
                clusters[i] = np.array([euclidean(data[i] - centers[j]) for j in range(k)])
            elif distance == 'm':
                clusters[i] = np.array([cityblock(data[i], centers[j]) for j in range(k)])
            elif distance == 'h':
                clusters[i] = np.array([hamming(data[i], centers[j]) for j in range(k)])
            else:
                raise ValueError('Unrecognized distance')
        clusters = np.argmin(clusters, axis=1)
        
        for i in range(k):
            centers[i] = np.mean(data[np.where(clusters == i)], axis=0)
        
        changed = not np.intersect1d(prev_centers, centers).size == centers.size
        centers_steps.append(centers.tolist())
        
    return centers, centers_steps