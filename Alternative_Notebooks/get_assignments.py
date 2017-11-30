from train_codebook import train_codebook
from get_local_features import get_local_features
from scipy.cluster.vq import vq, whiten
import matplotlib.pyplot as plt
import sys
import os.path as path

#dir = sys.path.insert(0,'./home/PycharmProjects/GDSA/Projecte/')
dir = path.dirname(__file__)

#dir = sys.path.insert(0,__file__)


def get_assignments(codebook, descriptors):

    #norm_descriptores = whiten(descriptores) # Normaliza descriptores
        #Con KMeans
    #assignments,_ = vq(descriptores, codebook)

    #Con MiniBatchKMeans
    assignments= codebook.predict(descriptors)
    return assignments


if __name__== "__main__":

    descriptor1 = get_local_features("TerrassaBuildings900/train/images/aaeoeolbth.jpg")
    codebook = train_codebook(5, descriptor1)
    descriptor2 = get_local_features("TerrassaBuildings900/val/images/aalfirydrf.jpg")
    assig = get_assignments(codebook, descriptor2)

    print(assig)
    print "Longitud del assignments= " + str(len(assig))
    

