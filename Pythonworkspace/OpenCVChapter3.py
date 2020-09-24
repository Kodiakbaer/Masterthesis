import cv2
import numpy as np

img = cv2.imread("Resources/RioAmAst.png")
print(img.shape)

imgResize = cv2.resize(img, (3000, 2000))
print(imgResize.shape)

imgCropped = img[0:500,200:500]

cv2.imshow("Rio", img)
# cv2.imshow("Rio resize", imgResize)
cv2.imshow("Rio cropped", imgCropped)
cv2.waitKey(0)