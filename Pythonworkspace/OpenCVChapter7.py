import cv2
import numpy as np


def empty(a):
    pass


imPath = "Resources/Stifte.PNG"
img = cv2.imread(imPath)
imgResize = cv2.resize(img, (640, 480))
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640,240)
cv2.createTrackbar("Hue Min", "TrackBars", 90, 255, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 115, 255, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 230, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 25, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 200, 255, empty)

while True:

    imgHSV = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")

    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")

    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(imgResize, imgResize, mask=mask)
    imgVer = np.vstack((imgResize, imgResult))

    #cv2.imshow("Original Image", img)
    #cv2.imshow("HSV Image", imgHSV)
    #cv2.imshow("Mask Image", mask)
    #cv2.imshow("Result Image", imgResult)
    cv2.imshow("Results", imgVer)
    cv2.imshow("Maske", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break