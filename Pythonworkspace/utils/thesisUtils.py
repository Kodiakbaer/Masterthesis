
from absl import app, flags, logging
from skimage.metrics import structural_similarity
import numpy as np
import cv2
import logging


def compareToBaselineImg(baseline, img, rect):
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
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(baseline, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    return filled_after             #
# Compares Images only in the Area inside the rectangle


def pixToPercent(rectPix, img):
    w, h = img.shape[1], img.shape[0]
    rectPrct = np.array([rectPix[0]/w, rectPix[1]/h, rectPix[2]/w, rectPix[3]/h])
    return rectPrct
# Converts Rectangle coordinates from Pixels to Percentage os the respective Image


def SaltPepperNoise(edgeImg):

    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 50:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)
# Method used in extraction Method


def findSignificantContour(edgeImg):
    image, contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
        # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)# From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])

    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    #print(largestContour)
    return largestContour
# Method used in extraction Method


def exctraction(img, rect = None):

    if type(img) is str and rect:
        image_vec = cv2.imread(str(img), 1)
        image_vec = image_vec[rect[1]:rect[3], rect[0]:rect[2]]
        logging.info('reading file: {}'.format(img))
    elif type(img) is str and not rect:
        image_vec = cv2.imread(str(img), 1)
        logging.info('reading file: {}'.format(img))
    elif rect is not None:
        image_vec = img[rect[1]:rect[3], rect[0]:rect[2]]
    else:
        image_vec = img

    g_blurred = cv2.GaussianBlur(image_vec, (3, 3), 0)

    if g_blurred is None:
        return

    blurred_float = g_blurred.astype(np.float32) / 255.0
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml.gz")
    edges = edgeDetector.detectEdges(blurred_float) * 255.0
    # cv2.imwrite('edge-raw.jpg', edges)

    edges_ = np.asarray(edges, np.uint8)
    SaltPepperNoise(edges_)
    # cv2.imwrite('edge.jpg', edges_)

    # image_display('edge.jpg')
    # print(sum(sum(edges_)))

    if sum(sum(edges_)) < 10e-16:
        return

    contour = findSignificantContour(edges_)
    # Draw the contour on the original image
    #contourImg = np.copy(image_vec)
    #cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    # cv2.imwrite('contour.jpg', contourImg)

    # image_display('contour.jpg')

    mask = np.zeros_like(edges_)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((1, 1), np.uint8), iterations=20)
    # mark inital mask as "probably background"
    # and mapFg as sure foreground

    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    # trimap_print[trimap_print == cv2.GC_PR_BGD] = 255
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    # cv2.imwrite('trimap.png', trimap_print)

    # cv2.imshow('mask', trimap_print)

    finalmask = cv2.cvtColor(trimap_print, cv2.COLOR_GRAY2BGR)
    im = img

    # final_im = finalmask * im
    im_thresh_color = cv2.bitwise_and(im, finalmask)
    if rect is not None:
        img[rect[1]:rect[3], rect[0]:rect[2]] = im_thresh_color
        return img
    else:
        return im_thresh_color
# Method for Image Segmentation


def overlapRect(rect1, rect2):
    overlap = np.zeros(4)
    r1x1, r1y1, r1x2, r1y2 = np.array(rect1[0:4])
    r2x1, r2y1, r2x2, r2y2 = np.array(rect2[0:4])


    if r1x1 > r2x1 and r1x1 < r2x2:
        overlap[0] = r1x1
    elif r2x1 > r1x1 and r2x1 < r1x2:
        overlap[0] = r2x1

    if r1y1 > r2y1 and r1y1 < r2y2:
        overlap[1] = r1y1
    elif r2y1 > r1y1 and r2y1 < r1y2:
        overlap[1] = r2y1

    if r1x2 > r2x1 and r1x2 < r2x2:
        overlap[2] = r1x2
    elif r2x2 > r1x1 and r2x2 < r1x2:
        overlap[2] = r2x2

    if r1y2 > r2y1 and r1y2 < r2y2:
        overlap[3] = r1y2
    elif r2y2 > r1y1 and r2y2 < r1y2:
        overlap[3] = r2y2

    return np.array(overlap)
# Calculation of overlap between to rectangles


def drawRect(rect, img, color, lineThickness):
    wh = np.flip(img.shape[0:2])
    x1y1 = tuple((np.array(rect[0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(rect[2:4]) * wh).astype(np.int32))

    draw = cv2.rectangle(img, x1y1, x2y2, color, lineThickness)
    return draw
# Draw rect on img

def rect_area(rect):
    x1, y1, x2, y2 = np.array(rect1[0:4])
    area = abs((x2 - x1) * (y2 - y1))
    return area


