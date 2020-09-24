import cv2
import numpy as np
img = cv2.imread("Resources/super-mario-wien.png")


# imgHor = np.hstack((img, img))
# imgVer = np.vstack((img, img))
# imgHorVer = np.hstack((imgVer, imgVer))
#
# #cv2.imshow("Horizontal", imgHor)
# #cv2.imshow("Vertical", imgVer)
# cv2.imshow("Horizontal", imgHorVer)


cv2.waitKey(0)