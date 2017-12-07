from get_params import get_params
import os
import pickle
import numpy as np
from sklearn.preprocessing import normalize

def bow(assignments,km):

    # Initialize empty descriptor of the same length as the number of clusters
    descriptor = np.zeros(np.shape(km.cluster_centers_)[0])

    # Build vector of repetitions
    for a in assignments:

        descriptor[a] += 1

    # L2 normalize
    descriptor = normalize(descriptor)

    return descriptor