from skimage.metrics import structural_similarity
import cv2
import numpy as np

before = cv2.imread("D:\MMichenthaler\HandOverHold\In\VideoTesting/frame1020.jpg")
after = cv2.imread("D:\MMichenthaler\HandOverHold\In\VideoTesting/frame1050.jpg")

rect = np.array([577, 1031, 623, 1074])

def compareToBaselineImg (baseline, img, rect):
    # Convert images to grayscale
    before_gray = cv2.cvtColor(baseline[rect[1]:rect[3], rect[0]:rect[2]], cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(img[rect[1]:rect[3], rect[0]:rect[2]], cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(baseline.shape, dtype='uint8')
    filled_after = (img[rect[1]:rect[3], rect[0]:rect[2]])

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(baseline, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    return filled_after


filled_after = compareToBaselineImg(before, after, rect)
# cv2.imshow('before', before)
# cv2.imshow('after', after)
# cv2.imshow('diff',diff)
# cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)
cv2.waitKey(0)