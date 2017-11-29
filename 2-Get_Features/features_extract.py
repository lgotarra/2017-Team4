# -*- coding: cp1252 -*-
from get_assignments import get_assignments
from train_codebook import train_codebook
from get_local_features import get_local_features
from build_bow import build_bow
from sklearn.preprocessing import normalize

import numpy as np
import warnings
import sys
import pickle
import os.path
import cv2

warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/oscarlinux/Escritorio/UPC/Q5/GDSA/Projecte/')
path_imagenes_train = "TerrassaBuildings900/train/images" #Directori on hi ha les imatges d'entrenament
path_imagenes_val = "TerrassaBuildings900/val/images" #Directori on hi ha les imatges de validació

def feature_extract (db_train, db_val,valortrain):
    if valortrain == 'train':
        fitxer=db_train
    else:
        fitxer=db_val
        
    with open(fitxer,'r') as f:
        image_list=f.readlines()
       # Inicialitzem el diccionari features
    features={}
    #num_clusters
    n_clusters=1024
    # Si guardessim el codebook el carragariem aqui
    # km = pickle.load(open("txt/codebook.cb","rb"))
    
    for image_name in image_list:
        image_name=image_name.replace('\n','')
        # Ara llegim l'imatge 
        nom=path_imagenes_train+'/'+image_name+'.jpg'
        feats=get_local_features(nom)
        if feats is not None:
            if False:
                feats=normalize(feats)                
                codebook=train_codebook(n_clusters, feats)
                #Calculem els assignments
                assignments=get_assignments(codebook,feats)
                #Calculem el Bag of Words (BoW)
                feats=build_bow(assignments,codebook)
            else:
            #Si els features estan buits
                feats=np.zeros(n_clusters)
            #Afegim l'entrada al diccionari
                features[image_name]=feats
            # Creem un directori a on guardarem el diccionari
            if valortrain== "train":
                pickle.dump(features, open("txt/bow_train.p", "wb" ))
                print ("Diccionari train creat \n" )
            else:
                
                pickle.dump(features, open("txt/bow_val.p", "wb" ) )
                print ("Diccionari validació creat \n")
                

feature_extract("txt/ID_images_train.txt", "txt/ID_images_val.txt", "train")