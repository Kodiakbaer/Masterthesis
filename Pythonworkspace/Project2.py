import cv2
import numpy as np


#############################
widthImg = 480
heightImg = 640

#############################

# This is a document scanner

cap = cv2.VideoCapture(0)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10, 150)


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]   # das -2 am ende wegen komischem Value Fehler
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest
            # print(len(approx))
            # objCor = len(approx)
            # x, y, w, h = cv2.boundingRect(approx)


def reorder (myPoints):

    # if myPoints.size  == 0:
    #     myPoints = np.zeros((4, 1, 2), np.int32)

    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add =myPoints.sum(1)

    #print("add", add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    #print("New Points", myPointsNew)

    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOut = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOut[20:imgOut.shape[0]-20, 20:imgOut.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))
    return imgOut


while True:
    success, img = cap.read()
    cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    # print(biggest)
    if biggest.size != 0:
        imgWarp = getWarp(img, biggest)


    cv2.imshow("Result", imgWarp)
    cv2.imshow("CameraFeed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break