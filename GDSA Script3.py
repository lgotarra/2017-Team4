#PROGRAMA AMB ORB

import cv2


img=cv2.imread('wifi.jpg',0) #Llegim l'imatge

orb=cv2.ORB_create() # Inicialitzem ORB detector

kp = orb.detect(img,None) # Trobem els Keypoints

kp,des = orb.compute(img,kp) # Compute els descriptors amb ORB

img2= cv2.drawKeypoints(img,kp,None,color=(0,255,0),flags=0)

"""
cv2.namedWindow('imatge', cv2.WINDOW_NORMAL)
cv2.imshow('imatge',img2) # mostrem l'imatge
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#Matching

img2=cv2.imread('wifi2.jpg',0) #Llegim l'imatge amb la qual farem matching


kp2,des2=orb.detectAndCompute(img2,None) #busquem els keypoints i els descriptors

bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True) #Creem l'objecte BFMatcher
matches=bf.match(des,des2) # Relacionem els descriptors
matches=sorted(matches,key=lambda x:x.distance) # Els ordenem a traves de la seva distància.


img3= cv2.drawMatches(img,kp,img2,kp2,matches[:10],None,flags=2) #Dibuixem les 10 coincidencies

cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
cv2.imshow('Matches',img3) # mostrem l'imatge
cv2.waitKey(0)
cv2.destroyAllWindows()