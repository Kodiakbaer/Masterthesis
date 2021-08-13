import time
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path
from skimage.metrics import structural_similarity
import numpy as np
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.utils import draw_persons
from utils.thesisUtils import *

imPath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1000.jpg"
img = cv2.imread(imPath)

rect = np.array([584, 1036, 625, 1069])
color = [0, 24, 27, 178, 190, 146]
roi = np.array(img[rect[1]:rect[3], rect[0]:rect[2]])

extractRoi = extraction(roi)
maskedRoi = mask_colour(extractRoi, color)

cv2.imshow("test", extractRoi)
cv2.waitKey(0)