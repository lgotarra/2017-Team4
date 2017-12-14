from get_params import get_params
import os, time
import numpy as np
import pickle
import cv2
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
def get_assignments(km,descriptors):

    assignments = km.predict(descriptors)

    return assignments
