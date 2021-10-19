import time
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path
from skimage.metrics import structural_similarity
import numpy as np
import csv
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.utils import draw_persons
from utils.thesisUtils import *


imPath = "D:/MMichenthaler/HandOverHold/NewDataVideo1Numbered/PhotoNr_0.jpg"
imPath2 = "D:/MMichenthaler/HandOverHold/NewDataVideo1Numbered/PhotoNr_120.jpg"
img = cv2.imread(imPath)
img2 = cv2.imread(imPath2)
#img = tf.expand_dims(img, 0)
rect = np.array([843, 2692, 992, 2835])
color = [0, 131, 93, 190, 255, 190]

#roi = np.array(img[rect[1]:rect[3], rect[0]:rect[2]])



#extractRoi = extraction(roi)

img1Masked = mask_colour(img, color)
img2Masked = mask_colour(img2, color)

allDiff, score = compare_baseline(img1Masked, img2Masked, rect)

cv2.imshow("test1", img1Masked)
cv2.imshow("test2", img2Masked)

cv2.imshow("test3", allDiff)
cv2.waitKey(0)


