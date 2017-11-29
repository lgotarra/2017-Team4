#PROGRAMA AMB ORB

import cv2


img=cv2.imread('escola_enginyeria_101.jpg',0) #Llegim l'imatge

orb=cv2.ORB_create() # Inicialitzem ORB detector

kp = orb.detect(img,None) # Trobem els Keypoints

kp,des = orb.compute(img,kp) # Compute els descriptors amb ORB

img2= cv2.drawKeypoints(img,kp,None,color=(0,255,0),flags=0)


cv2.namedWindow('imatge', cv2.WINDOW_NORMAL)
cv2.imshow('imatge',img2) # mostrem l'imatge
cv2.waitKey(0)
cv2.destroyAllWindows()

