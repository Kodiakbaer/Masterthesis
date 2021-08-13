
from absl import app, flags, logging
from skimage.metrics import structural_similarity
import numpy as np
import cv2
import logging


def compare_baseline(baseline, img, rect=None):
    if rect is None:
        rect = [0, 0, baseline.shape[1], baseline.shape[0]]
    newBase = baseline.copy()
    newImg = img.copy()
    # Convert images to grayscale
    before_gray = cv2.cvtColor(newBase[rect[1]:rect[3], rect[0]:rect[2]], cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(newImg[rect[1]:rect[3], rect[0]:rect[2]], cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    # print("Image similarity", score)

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

    mask = np.zeros(newBase.shape, dtype='uint8')
    filled_after = (newImg[rect[1]:rect[3], rect[0]:rect[2]])

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x, y, w, h = cv2.boundingRect(c)
            #cv2.rectangle(newBase, (x, y), (x + w, y + h), (36, 255, 12), 2)
            #cv2.rectangle(newImg, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    return filled_after, score
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


def extraction(img, rect = None):

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
    x1, y1, x2, y2 = np.array(rect[0:4])
    area = abs((x2 - x1) * (y2 - y1))
    return area
# calculate area of rectangle


def mask_colour(img, colorRange):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #for color in myColors:
    lower = np.array(colorRange[0:3])
    upper = np.array(colorRange[3:6])

    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    return imgResult
# Method to mask image by color


def mark_rectangle(action, x, y, flags, *userdata) :
  # Referencing global variables
  global top_left_corner, bottom_right_corner, tlc, brc
  # Mark the top left corner when left mouse button is pressed
  if action == cv2.EVENT_LBUTTONDOWN:
    top_left_corner = [(x,y)]
    # When left mouse button is released, mark bottom right corner
    tlc = [x,y]
    print(tlc)
  elif action == cv2.EVENT_LBUTTONUP:
    bottom_right_corner = [(x,y)]
    # Draw the rectangle
    cv2.rectangle(tempBase, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
    cv2.imshow("Window", tempBase)
    brc = [x,y]
    print(brc)
    rect = [tlc[0],tlc[1],brc[0],brc[1]]
    print(rect)
    tempHolds.append(rect)
#method for marking a rectangle in a window and


def hold_marker(image):
    #image = cv2.imread("D:\MMichenthaler\VideoFrames\Video2\Video2_frame1000.jpg")
    # Make a temporary image, will be useful to clear the drawing
    temp = image.copy()
    global tempBase
    global tempHolds
    tempBase = image.copy()
    tempHolds = []
    # Create a named window

    cv2.namedWindow("Window")
    # highgui function called when mouse events occur
    cv2.setMouseCallback("Window", mark_rectangle, tempHolds)

    k = 0
    # Close the window when key q is pressed
    while k != 113:
        # Display the image
        cv2.imshow("Window", image)
        k = cv2.waitKey(0)
        # If c is pressed, clear the window, using the dummy image
        if (k == 99):
            image = temp.copy()
            cv2.imshow("Window", image)
            tempHolds = []
        if (k == 32):
            print(tempHolds)
        if (k == 115):
            f = open("holds.txt", "w")
            f.write(str(tempHolds))

    cv2.destroyAllWindows()
    del tempBase
    holds = tempHolds
    del tempHolds
    return holds