# -*- coding: utf-8 -*-
import cv2
import matplotlib as plt
import sys

sys.path.insert(0,'/home/oscarlinux/Escritorio/UPC/Q5/GDSA/Projecte/')

# Retorna els descriptors d'una imatge
def get_local_features(imatge):
    
    # Altres possibles mètodes serien SURF o ORB, fins i tot el FASTSIFT.
    img = cv2.imread(imatge)
    sift = cv2.SIFT()
    kp, des =sift.detectAndCompute(img, None) #kp= número de punts d'interes -- des= descriptors
    #kp, des = orb.detectAndCompute(small,None) #kp= número de punts d'interes -- des= descriptors
    print("S'han creat: " + str(len(des)) + " descriptors per l'imatge " + "\"" + imatge + "\"" + " amb "  + str(len(des[4])) + " Keypoints per descriptor.")
    return des

if __name__ == "__main__":
    # Aixó és per ensenyar que funciona
    image= "TerrassaBuildings900/train/images/aaeoeolbth.jpg"
    features= get_local_features(image)
    num_descript = len(features)
    key_points = len(features[4]) # tamaño de un vector del descriptor