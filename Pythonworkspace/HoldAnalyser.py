import time
import os
from timeit import default_timer as timer
#from datetime import datetime
import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
from utils.thesisUtils import *
from pathlib import Path
#from skimage.metrics import structural_similarity      #--------- beschreiben/behirnen
import numpy as np
import cv2                                             #--------- beschreiben
import csv


flags.DEFINE_string('CSVpath', ' ', 'path where the file containing the holds will be stored')

badHolds = [4, 5, 6, 8, 9, 10, 11, 14, 18, 19]
def main(_argv):
    if os.path.isfile(FLAGS.CSVpath):
        with open(FLAGS.CSVpath, "r") as file:
            holdsSt = []
            stHolds = list(csv.reader(file, delimiter=','))
            # print(stHolds)
            for elem in stHolds:
                for elem2 in elem:
                    elem3 = elem2.replace('[', '').replace(']', '')
                    holdsSt.append(elem3.split(','))

            holds = [list(map(int, rec)) for rec in holdsSt]
            logging.info('holds loaded')
    print(holds)
    print("bad holds")
    for holdID in range(len(holds)):  # alle hände mit allen griffen verschneiden
        if holds[holdID][4]-1 in badHolds:
            print("Hold Nr: " + str(holds[holdID][4]-1) + " area: " + str(rect_area(holds[holdID])))

    print("good holds")
    for holdID in range(len(holds)):
        if holds[holdID][4]-1 not in badHolds:
            print("Hold Nr: " + str(holds[holdID][4] - 1) + " area: " + str(rect_area(holds[holdID])))



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass