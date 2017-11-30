# -*- coding: utf-8 -*-
import cv2
import matplotlib as plt
import sys
import os.path as path

sys.path.insert(0,'./home/PycharmProjects/GDSA/')

#dir = path.dirname(__file__)
path_train = "../TB2016/train/images"
path_val = "../TB2016/val/images"
path_txt = "../txt/"
sys.path.insert(0,dir)

# Retorna els descriptors d'una imatge
def get_local_features(imatge):
    
    # Altres possibles mètodes serien SURF o ORB, fins i tot el FASTSIFT.
    img = cv2.imread(imatge)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des =sift.detectAndCompute(img, None) #kp= número de punts d'interes -- des= descriptors
    #kp, des = orb.detectAndCompute(small,None) #kp= número de punts d'interes -- des= descriptors
    print("S'han creat: " + str(len(des)) + " descriptors per l'imatge " + "\"" + imatge + "\"" + " amb "  + str(len(des[4])) + " Keypoints per descriptor.")
    return des

if __name__ == "__main__":
    # Aixó és per ensenyar que funciona
    image= path.join(path_train,"../aaeoeolbth.jpg")
    features= get_local_features(image)
    num_descript = len(features)
    key_points = len(features[4]) # tamaño de un vector del descriptor