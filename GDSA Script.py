# PROGRAMA SIFT

import cv2
import numpy as np

img=cv2.imread('wifi.jpg') # Llegim l'imatge que esta en el mateix directori i la posem a la variable img

sift=cv2.xfeatures2d.SIFT_create() #inicialitzem el sift.
kp,des= sift.detectAndCompute(img,None) # trobem els keypoints i els descriptors

cv2.drawKeypoints(img,kp,img,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Dibuixem els keypoints a la img i la seva orientació

"""cv2.namedWindow('imatge', cv2.WINDOW_NORMAL)  # mostrem l'imatge
cv2.imshow('imatge',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#Match amb unaltre foto

img2=cv2.imread('wifi2.jpg') # Llegim l'imatge amb la que farem match

kp2,des2= sift.detectAndCompute(img2,None) #Busquem els keypoints i els descriptors de la segona imatge

bf=cv2.BFMatcher()
matches=bf.knnMatch(des,des2,k=2)

good=[]
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3=cv2.drawMatchesKnn(img,kp,img2,kp2,good,None,flags=2)
cv2.namedWindow('imatge', cv2.WINDOW_NORMAL)  # mostrem l'imatge
cv2.imshow('imatge',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
