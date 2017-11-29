# -*- coding: utf-8 -*-
from sklearn.cluster import MiniBatchKMeans
from get_local_features import get_local_features
import sys

sys.path.insert(0,'/home/oscarlinux/Escritorio/UPC/Q5/GDSA/Projecte/')


def train_codebook(numClusters, descriptores): #Només para las imatges de train

    """Amb KMeans
    centroides,_= kmeans(descriptores, numClusters)"""

    #Amb MiniBatchKMeans (És més ràpid i més eficient que KMeans)
    centroides= MiniBatchKMeans(numClusters)
    centroides.fit(descriptores)

    return centroides # retorna el vector codebook


if __name__== "__main__":
    # Es per comprovar que la funció és correcta
    descriptors = get_local_features("TerrassaBuildings900/train/images/aaeoeolbth.jpg")
    codebook1 = train_codebook(1, descriptors)