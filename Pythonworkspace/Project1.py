import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

myColors = [[90, 230, 25, 115, 255, 200],  # Grün 87 112 232 255 24 200
            [125, 155, 20, 140, 255, 145],  # Violett 124 138 155 225 20 146
            [0, 195, 155, 15, 255, 255]]  # Orange 0 15 110 255 200 255

0, 15, 195, 255, 155, 255

def findColour(img, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        cv2.imshow(str(color[0]),mask)



while True:
    success, img = cap.read()
    findColour(img, myColors)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break