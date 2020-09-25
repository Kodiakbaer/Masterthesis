import cv2
import numpy as np

frameWidth = 1240
frameHeight = 960
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

myColors = [[76, 111, 88, 92, 255, 233],  # Grün 76 92 111 255 88 233
            [121, 103, 80, 141, 255, 255],  # Violett 121 141 103 255 80 255
            [0, 153, 188, 15, 255, 255]]  # Orange 0 15 153 255 188 255

myColorValues = [[109, 210, 46],     # BGR Fromat
                 [210, 38, 138],
                 [40, 170, 255]]


myPoints = []  # [x, y, colorId]

def findColour(img, myColors, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        # cv2.imshow(str(color[0]),mask)
        x, y = getContours(mask)
        cv2.circle(imgResult, (x, y), 10, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count +=1
    return newPoints


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]   # das -2 am ende wegen komischem Value Fehler
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y


def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)



while True:
    success, img = cap.read()
    imgResult = img.copy()
    newPoints = findColour(img, myColors, myColorValues)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)


    cv2.imshow("Video", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

