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


imPath = "D:/MMichenthaler/HandOverHold/NewDataVideo11Numbered/PhotoNr_645.jpg"
imPath2 = "D:/MMichenthaler/HandOverHold/NewDataVideo11Numbered/PhotoNr_765.jpg"
img1 = cv2.imread(imPath)
img2 = cv2.imread(imPath2)
#img = tf.expand_dims(img, 0)
rect = np.array([1081, 2832, 1147, 2901, 6])
color = [120,0,41,174,133,225]

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
#label = tk.Label(None, text='Always pick the top left Corner first and then the bottom right one', font=('Times', '18'),
#                 fg='black')
#label.pack()
#label.mainloop()

img1Masked = mask_colour(img1, color)
#print(img1Masked)
img2Masked = mask_colour(img2, color)
#cv2.imshow("test1", img1Masked)
#cv2.imshow("test2", img2Masked)
allDiff, score, scorePix = compare_baseline(img1Masked, img2Masked, rect)
#allDiff = cv2.resize(allDiff, (725, 1288), interpolation=cv2.INTER_AREA)
cv2.imshow("All Differences via greyascale", allDiff)
print(str(allDiff.shape[1] * allDiff.shape[0]))
print('Fremdpixel: ' + str((allDiff.shape[1] * allDiff.shape[0])-scorePix))
print('NonZero Pixel: ' + str(cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
print('overlap: ' + str((allDiff.shape[1] * allDiff.shape[0]-scorePix)/cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
print('similiarity: ' + str(score))




#allDiffHSV, score2, scorePix2 = compare_baseline_HSV(img, img2, rect)

#allDiffHSV = cv2.resize(allDiffHSV, (725, 1288), interpolation=cv2.INTER_AREA)
#cv2.imshow("Hue Differences", allDiffHSV)
#print(str(allDiffHSV.shape[1] * allDiffHSV.shape[0]))
#print(scorePix2)
cv2.waitKey(0)

