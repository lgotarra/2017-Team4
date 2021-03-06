from get_assignments import get_assignments
import matplotlib.pyplot as plt
from train_codebook import train_codebook
from get_local_features import get_local_features
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import sys
from sklearn.preprocessing import normalize, StandardScaler
import os.path as path

#dir = sys.path.insert(0,'./home/PycharmProjects/GDSA/Projecte/')
dir = path.dirname(__file__)

#dir = sys.path.insert(0,__file__)
terrassa_buildings = '../TB2016/'



def build_bow(assignments, n):
    # Inicialitzem a zeros un vector de mida dels clusters
    descriptor = np.zeros((n,))

    # Construim un vector de repeticions.Cada assignments li atribuim un cluster
    for n_assig in assignments:
        descriptor[n_assig] += 1

    # L2 normalize
    descriptor = normalize(descriptor)

    return descriptor


# Comprovem que funciona
descriptor1 = get_local_features(path.join(terrassa_buildings, "../train/images/aaeoeolbth.jpg"))
codebook = train_codebook(5, descriptor1)
descriptor2 = get_local_features(path.join(terrassa_buildings, "../val/images/aalfirydrf.jpg"))
assig = get_assignments(codebook, descriptor2)

# Crea un vector ordenat amb els descriptors que equival a cada regio (k=5)
asdf = build_bow(assig, 50)
print asdf
print ("Numero de regiones diferentes: " + str(len(asdf)))