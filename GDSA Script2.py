#PROGRAMA SURF

import cv2

img=cv2.imread('wifi.jpg',0) #Llegim l'imatge


surf =  cv2.xfeatures2d.SURF_create(400) # Creem un objecte SURF. El valor de 400 és el llindar

kp,des=surf.detectAndCompute(img,None) # trobem els keypoints i els descriptors

img2=cv2.drawKeypoints(img,kp,None,(0,0,255),4) # Dibuixem els keypoints a la img2

cv2.namedWindow('imatge', cv2.WINDOW_NORMAL)
cv2.imshow('imatge',img2) # mostrem l'imatge
cv2.waitKey(0)
cv2.destroyAllWindows()
