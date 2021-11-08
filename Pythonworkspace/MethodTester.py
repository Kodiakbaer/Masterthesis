import time
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path
from skimage.metrics import structural_similarity
import numpy as np
import csv
import cv2
import tkinter as tk
from tkinter import simpledialog

import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.utils import draw_persons
from utils.thesisUtils import *


imPath = "D:/MMichenthaler/HandOverHold/NewDataVideo4Numbered/PhotoNr_9570.jpg"
imPath2 = "D:/MMichenthaler/HandOverHold/NewDataVideo4Numbered/PhotoNr_9690.jpg"
img1 = cv2.imread(imPath)
img2 = cv2.imread(imPath2)
#img = tf.expand_dims(img, 0)
rect = np.array([436, 39, 463, 66, 40])
color = [26,0,0,103,255,160]

#roi = np.array(img[rect[1]:rect[3], rect[0]:rect[2]])


'''
ROOT = tk.Tk()

ROOT.withdraw()
# the input dialog
USER_INP = simpledialog.askstring(title="Score",
                                  prompt="Put in the Score for the Hold:")

# check it out
print(int(USER_INP))

#extractRoi = extraction(roi)


allDiff, score = compare_baseline(img1Masked, img2Masked, rect)

cv2.imshow("test1", img1Masked)
cv2.imshow("test2", img2Masked)
'''
#img1HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#img2HSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)


img1Masked = mask_colour(img1, color)
#print(img1Masked)
img2Masked = mask_colour(img2, color)
cv2.imshow("test1", img1Masked)
cv2.imshow("test2", img2Masked)
allDiff, score, scorePix = compare_baseline(img1, img2,rect)
#allDiff = cv2.resize(allDiff, (725, 1288), interpolation=cv2.INTER_AREA)
cv2.imshow("All Differences via greyascale", allDiff)
print(str(allDiff.shape[1] * allDiff.shape[0]))
print('Fremdpixel: ' + str((allDiff.shape[1] * allDiff.shape[0])-scorePix))
print('NonZero Pixel: ' + str(cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
print((allDiff.shape[1] * allDiff.shape[0]-scorePix)/cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY)))
print(score)




#allDiffHSV, score2, scorePix2 = compare_baseline_HSV(img, img2, rect)

#allDiffHSV = cv2.resize(allDiffHSV, (725, 1288), interpolation=cv2.INTER_AREA)
#cv2.imshow("Hue Differences", allDiffHSV)
#print(str(allDiffHSV.shape[1] * allDiffHSV.shape[0]))
#print(scorePix2)
cv2.waitKey(0)

