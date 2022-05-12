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


from utils.thesisUtils import *


imPath = "D:/MMichenthaler/HandOverHold/NewDataVideo7Numbered/PhotoNr_2380.jpg"
imPath2 = "D:/MMichenthaler/HandOverHold/NewDataVideo7Numbered/PhotoNr_2260.jpg"
img1 = cv2.imread(imPath)
img2 = cv2.imread(imPath2)
#img = tf.expand_dims(img, 0)
rect = np.array([1537, 1318, 1600, 1407, 24])
color = [100,24,0,255,114,255]

img1Masked = mask_colour(img1, color)
#print(img1Masked)
img2Masked = mask_colour(img2, color)
#cv2.imshow("test1", img1Masked)
#cv2.imshow("test2", img2Masked)
allDiff, score, scorePix = compare_baseline(img1Masked, img2Masked, rect)
#allDiff = cv2.resize(allDiff, (725, 1288), interpolation=cv2.INTER_AREA)
imgRect = img1Masked[rect[1]:rect[3], rect[0]:rect[2]]
imgRect2 = img2Masked[rect[1]:rect[3], rect[0]:rect[2]]
imgRect3 = img1[rect[1]:rect[3], rect[0]:rect[2]]
imgRect4 = img2[rect[1]:rect[3], rect[0]:rect[2]]
cv2.imwrite('D:/MMichenthaler/SSIMrelations/img'+str(rect[4])+'.png', imgRect)
cv2.imwrite('D:/MMichenthaler/SSIMrelations/img'+str(rect[4])+'_base.png', imgRect2)
cv2.imwrite('D:/MMichenthaler/SSIMrelations/img'+str(rect[4])+'_raw.png', imgRect3)
cv2.imwrite('D:/MMichenthaler/SSIMrelations/img'+str(rect[4])+'_raw_base.png', imgRect4)
cv2.imwrite('D:/MMichenthaler/SSIMrelations/img'+str(rect[4])+'_diff.png', allDiff)
#cv2.imwrite('D:/MMichenthaler/SSIMrelations/img'+str(rect[4])+'_diff_raw.png', diff)
#cv2.imshow("All Differences via greyascale", allDiff)
print('Hold '+str(rect[4]))
print('Flaeche: '+str(allDiff.shape[1] * allDiff.shape[0]))
print('Fremdpixel: ' + str((allDiff.shape[1] * allDiff.shape[0])-scorePix))
print('NonZero Pixel: ' + str(cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
print('NonZero Pixel [%]: ' + str((cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))/(allDiff.shape[1] * allDiff.shape[0]))*100))
print('NonZero Pixel masked: ' + str(cv2.countNonZero(cv2.cvtColor(imgRect2, cv2.COLOR_BGR2GRAY))))
print('NonZero Pixel masked[%]: ' + str((cv2.countNonZero(cv2.cvtColor(imgRect2, cv2.COLOR_BGR2GRAY))/(imgRect2.shape[1] * imgRect2.shape[0]))*100))
print('hand approx: ' + str(cv2.countNonZero(cv2.cvtColor(imgRect2, cv2.COLOR_BGR2GRAY)) - cv2.countNonZero(cv2.cvtColor(imgRect, cv2.COLOR_BGR2GRAY))))
print('hand approx zu Gesamtflaeche[%]: '+ str((cv2.countNonZero(cv2.cvtColor(imgRect2, cv2.COLOR_BGR2GRAY)) - cv2.countNonZero(cv2.cvtColor(imgRect, cv2.COLOR_BGR2GRAY)))/(allDiff.shape[1] * allDiff.shape[0])*100))
print('overlap: ' + str((allDiff.shape[1] * allDiff.shape[0]-scorePix)/cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
print('similiarity: ' + str(score*100))
#cv2.waitKey(0)