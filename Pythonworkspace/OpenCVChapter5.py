import cv2
import numpy as np

img = cv2.imread("Resources/Cards.jpeg")

width, height = 900, 1490
pts1 = np.float32([[345,370],[872,366],[208,1321],[968,1330]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOut = cv2.warpPerspective(img, matrix, (width,height))


cv2.imshow("Image", img)
cv2.imshow("Output", imgOut)

cv2.waitKey(0)