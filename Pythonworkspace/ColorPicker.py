import cv2
import numpy as np
import csv

def empty(a):
    pass


imPath = "D:/MMichenthaler/Data_15-09-2021_Workspace/NewVideo1/NewVideo1_frame250.jpg"
colorPath = 'D:/MMichenthaler/HandOverHold/ScriptTestNewData2/'
img = cv2.imread(imPath)
imgResize = cv2.resize(img, (540, 960))
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640,240)

with open(colorPath + 'colors.csv', "r") as file:
    StColor = list(csv.reader(file, delimiter=','))
    colorInt = [list(map(int, rec)) for rec in StColor]
    colorInt = colorInt[0]
    print(colorInt)

    # print(stHolds)
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
    #print(h_min, h_max, s_min, s_max, v_min, v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(imgResize, imgResize, mask=mask)
    #imgVer = np.vstack((imgResize, imgResult))

    cv2.imshow("Original Image", imgResize)
    #cv2.imshow("HSV Image", imgHSV)
    #cv2.imshow("Mask Image", mask)
    #cv2.imshow("Result Image", imgResult)
    cv2.imshow("Results", imgResult)
    cv2.imshow("Maske", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open(colorPath + 'colors.csv', 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            lowerUpper = [h_min,s_min,v_min,h_max,s_max,v_max]
            # write a row to the csv file
            writer.writerow(lowerUpper)
        break