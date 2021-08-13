import cv2
import numpy as np

# Ebene Entzerrung
img = cv2.imread("Resources/LSW.jpg")

width, height = 2000, 2000
pts1 = np.float32([[710,240],[3270,290],[620,2820],[3295,2800]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOut = cv2.warpPerspective(img, matrix, (width,height))


cv2.imshow("Image", img)
cv2.imshow("Output", imgOut)

cv2.imwrite("LSW.jpg", imgOut)
cv2.waitKey(0)