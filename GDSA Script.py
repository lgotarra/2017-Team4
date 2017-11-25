# PROGRAMA SIFT
import cv2
import numpy as np

img=cv2.imread('wifi.jpg', 0)# Llegim l'imatge que esta en el mateix directori i la posem a la variable img

sift= cv2.xfeatures2d.SIFT_create(400)

kp,des= sift.detectAndCompute(img, None) # trobem els keypoints i els descriptors

# Dibuixem els keypoints a la img i la seva orientació

img1=cv2.drawKeypoints(img, kp, None, (255,0,0),4)

# mostrem l'imatge

cv2.namedWindow('imatg', cv2.WINDOW_NORMAL)
cv2.imshow('imatg', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


