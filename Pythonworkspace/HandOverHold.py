import time
import os
from timeit import default_timer as timer
#from datetime import datetime
import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path
from skimage.metrics import structural_similarity      #--------- beschreiben/behirnen
import numpy as np
import cv2                                             #--------- beschreiben
import csv
import tensorflow as tf
from yolov3_tf2.models import (                        #--------- beschreiben
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from yolov3_tf2.utils import draw_persons
from utils.thesisUtils import *

logging.basicConfig(level=logging.INFO)




########################################################################################################################
############################# First test of combining Methods to detect a gripped hold #################################
########################################################################################################################

'''                                             ____
                    \          /        |       |   \
                     \        /         |       |___/
                      \  /\  /          |       |
                       \/  \/           |       |
'''
##################### YOLO Parameter #################
flags.DEFINE_string('classes', './data/climber.names', 'path to classes file')
flags.DEFINE_string('weights', 'H:\MasterArbeit\yolov3-tf2\checkpoints/yolov3.tf',
                    'path to weights file')

flags.DEFINE_string('baseClasses', './data/hold.names', 'path to classes file')             # diese 3 Parameter beziehen sich auf die
flags.DEFINE_string('baseWeights', None,                                                    # detection der Griffe auf dem Baseline img
                    'path to weights file')

flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')                       # noch nicht damit experimentiert
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

####################### inputs #######################
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('imDir', None, 'path to input image directory')    # Video taken from the same Perspective as Baseline Image (here a directory with just the frames is used -> in a later iteration a direct video feed should be used
flags.DEFINE_string('vidFile', None, 'path to input Video')
# flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('baseline', None, 'Baseline Image of Wall without the climber')  # Baseline Image of Wall without the climber
flags.DEFINE_integer('delay', 120, 'Delay for comparison baseline')
flags.DEFINE_integer('frameReduction', 10, 'Reduction of used frames by the factor')
flags.DEFINE_float('similarity', 0.6, 'Similarity percent under which a Hold is declared gripped')
flags.DEFINE_float('holdOverlap', 0.9, 'Similarity percent under which a Hold is declared gripped')
flags.DEFINE_integer('handPix', 2000, 'approximate hand size in pixel')
flags.DEFINE_list('holdColor', None, 'if you already know your color values you can put them in here')


##################### conditions #####################
flags.DEFINE_boolean('detection', False, 'detection Yes or No')
flags.DEFINE_boolean('holdsDetection', False, 'Hold detection Yes or No')
flags.DEFINE_boolean('cam', False, 'is a cam used?')
flags.DEFINE_boolean('video', False, 'is a video as a file or cam feed used?')
flags.DEFINE_boolean('colorMask', False, 'do you want to apply a color mask for the baseline comaprison?')


###################### outputs ######################
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_string('holdsOut', ' ', 'path to output image')
flags.DEFINE_string('numberedSource', ' ', 'path to numbered images')
flags.DEFINE_string('CSVpath', ' ', 'path where the file containing the holds will be stored')
#----------------------for testing purposes
#basePath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1000.jpg"
#basePath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1921.jpg"
#basePath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1871.jpg"
#base = cv2.imread(basePath)

# baseRect = np.array([584, 1036, 625, 1069])
# color = [0, 131, 93, 190, 255, 190]
# aoi = np.array(base[baseRect[1]:baseRect[3], baseRect[0]:baseRect[2]])



#----------------------------------------------------------------------------------------------------------------------#

def main(_argv):
    start = timer()
    #timerStart = time.time()
    #basePath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1000.jpg"
    #base = cv2.imread(basePath)
    allClimbers = []
    climbersThisPic = []
    #holds = hold_marker(base)
    #print(holds)
    print(FLAGS.colorMask)
    #------------------------------------------
    # Dieser Part ist für die Detection mittels Yolo, da das nicht ausreichend funktioniert wird er hier nun bis auf 
    # weiteres auskommentiert gelassen
    #---------------------------------------------

    if FLAGS.baseline and FLAGS.holdsDetection is True:
        base_raw = tf.image.decode_image(
            open(FLAGS.baseLine, 'rb').read(), channels=3)
        base = tf.expand_dims(base_raw, 0)
        base = transform_images(base, FLAGS.size)

        t1 = time.time()
        baseBoxes, baseScores, BaseClasses, BaseNums = yolo(base)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        base = cv2.cvtColor(base_raw.numpy(), cv2.COLOR_RGB2BGR)
        base = draw_outputs(base, (baseBoxes, baseScores, BaseClasses, BaseNums), class_names)     # detection used on  Baseline Img
        cv2.imwrite(FLAGS.output + 'baselineImg.jpg', base)
        logging.info('Baseline set and saved to: {}'.format(FLAGS.output) + str(count))
    # using a separate detector for holds on the bare wall image to set a baseline and saving the results

    elif os.path.isfile(FLAGS.CSVpath + "holds.csv"):
        with open(FLAGS.CSVpath + "holds.csv", "r") as file:
            holdsSt = []
            stHolds = list(csv.reader(file, delimiter=','))
            # print(stHolds)
            for elem in stHolds:
                for elem2 in elem:
                    elem3 = elem2.replace('[', '').replace(']', '')
                    holdsSt.append(elem3.split(','))

            holds = [list(map(int, rec)) for rec in holdsSt]
            logging.info('holds loaded')

    elif FLAGS.baseline and FLAGS.CSVpath:                        # dieser code ist um die Griffe in einem Gui zu markieren
        base = cv2.imread(FLAGS.baseline)
        holds = hold_marker(base, FLAGS.CSVpath)
        print(holds)



    '''
    holds = [[843, 2692, 992, 2835],  # holds für newVideo1
             [712, 2516, 891, 2644],
             [879, 2409, 1061, 2519],
             [787, 2039, 924, 2132],
             [912, 1875, 1025, 1971],
             [775, 1795, 888, 1887],
             [1013, 1705, 1120, 1798],
             [819, 1392, 933, 1520],
             [1028, 1112, 1141, 1199],
             [849, 1079, 959, 1213],
             [807, 909, 941, 987],
             [956, 713, 1037, 799],
             [864, 602, 950, 689],
             [1022, 584, 1129, 671],
             [903, 408, 986, 495],
             [1010, 280, 1123, 367]]




    

             
    holds = [[555, 1253, 594, 1288],            # für weitere Testungen die mit hold_marker markierten Griffe von Video2 des alten datensatzes
             [588, 1178, 627, 1215],
             [584, 1107, 626, 1141],
             [579, 1035, 631, 1075],
             [584, 967, 618, 994],
             [545, 862, 599, 908],
             [524, 830, 570, 873],
             [487, 755, 565, 828],
             [512, 680, 584, 741],
             [526, 611, 597, 681],
             [561, 550, 617, 588],
             [532, 489, 586, 528],
             [622, 404, 667, 449],
             [585, 378, 616, 400],
             [531, 392, 565, 420],
             [523, 319, 565, 366],
             [468, 276, 503, 305],
             [531, 177, 579, 223],
             [452, 103, 495, 146]]
    

    holds = [[354, 1252, 447, 1323],        # für weitere Testungen die mit hold_marker markierten Griffe von Video2 des neuen Datensatzes
             [432, 1347, 492, 1397],
             [518, 1291, 569, 1349],
             [439, 1206, 528, 1260],
             [395, 1113, 470, 1161],
             [531, 1069, 579, 1105],
             [540, 988, 581, 1027],
             [396, 1023, 461, 1068],
             [459, 937, 513, 983],
             [321, 964, 389, 1033],
             [318, 845, 363, 899],
             [465, 766, 511, 820],
             [314, 772, 367, 809],
             [357, 700, 413, 740],
             [475, 636, 520, 675],
             [374, 612, 437, 679],
             [500, 600, 542, 627],
             [424, 543, 481, 602],
             [516, 560, 567, 596],
             [568, 488, 609, 520],
             [403, 456, 469, 493],
             [473, 424, 532, 462],
             [476, 358, 517, 399],
             [414, 382, 452, 424],
             [515, 295, 564, 334],
             [455, 327, 494, 366],
             [450, 206, 495, 245],
             [514, 235, 558, 273],
             [496, 103, 459, 144],
             [514, 180, 560, 216]]
    '''
    # auskommentieren weil flags colormask nicht funktioniert
    print(FLAGS.colorMask)
    if os.path.isfile(FLAGS.CSVpath + 'colors.csv') and FLAGS.colorMask is True:
        with open(FLAGS.CSVpath + 'colors.csv', "r") as file:
            StColor = list(csv.reader(file, delimiter=','))
            color = [list(map(int, rec)) for rec in StColor]
            color = color[0]
            logging.info('color loaded ' + str(color))


    elif FLAGS.colorMask is True:
        color = color_picker(FLAGS.baseline, FLAGS.CSVpath)

    if FLAGS.detection is True:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)

        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)

        yolo.load_weights(FLAGS.weights).expect_partial()
        logging.info('weights loaded')

        class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        logging.info('classes loaded')

        # if FLAGS.tfrecord:
        #     dataset = load_tfrecord_dataset(
        #         FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        #     dataset = dataset.shuffle(512)
        #     img_raw, _label = next(iter(dataset.take(1)))
        #     img = tf.expand_dims(img_raw, 0)
        #     img = transform_images(img, FLAGS.size)
        #
        #     t1 = time.time()
        #     boxes, scores, classes, nums = yolo(img)
        #     t2 = time.time()
        #     logging.info('time: {}'.format(t2 - t1))
        #
        #     logging.info('detections:')
        #     for i in range(nums[0]):
        #         logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
        #                                            np.array(scores[0][i]),
        #                                            np.array(boxes[0][i])))
        #         allhands.append(boxes[0][i])
        #
        #     img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        #     img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        #     cv2.imwrite(FLAGS.output, img)
        #     logging.info('output saved to: {}'.format(FLAGS.output))
        # print(sorted(os.listdir(FLAGS.imDir), key=lambda x: int(x[15:-5])))

        if FLAGS.imDir:                                                 # für Detection auf allen bildern in imDir
            climberCounter = 0
            makeNumbered = False
            #for count, dirImg in enumerate(Path(FLAGS.imDir).iterdir()):                    # , key=lambda x: int(x[69:-4]) beim key die anzahl der Zeichen des Paths angeben der vor der nummerierung steht
            for count, dirImg in enumerate(sorted(os.listdir(FLAGS.imDir), key=lambda x: int(x[16:-4]))): # ANPASSEN WENN UNTER ODER ÜBER 10 auf 16
                #climbDetect = False

                #img_raw = tf.image.decode_image(
                #   open(dirImg, 'rb').read(), channels=3)

                img_raw = cv2.imread(FLAGS.imDir + dirImg)
                img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, FLAGS.size)


                t1 = time.time()
                boxes, scores, classes, nums = yolo(img)
                t2 = time.time()
                logging.info('time: {}'.format(t2 - t1))

                logging.info('detections:')
                img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)

                for i in range(nums[0]):
                    #print(np.array(boxes[0][i]))
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                       np.array(scores[0][i]),
                                                       np.array(boxes[0][i])))
                    if class_names[int(classes[0][i])] == "person":


                        climbersThisPic.append(np.array(boxes[0][i]))                           # saving all detected climbers in allclimbers------ für testung auskommentieren PROBLEM mit 2 personen in bild

                if not os.listdir(FLAGS.numberedSource) or makeNumbered is True:
                    makeNumbered = True                                                                                        #if climbDetect is False:                            #+----
                    cv2.imwrite(FLAGS.numberedSource + 'PhotoNr_' + str(climberCounter) + '.jpg', img)       #| Mit neuem Datensatz einmal diesen block unkommentiert mitlaufen lassen
                    #DIESE ZEILEN FÜR NUMBERED                                                            #| dieser Block speichert jeden frame nummeriert ab, nicht nur jeden in dem eine person vorkommt,
                    climberCounter += 1                                                                      #| da sonst bei versagen des yolo keine möglichkeit besteht die gegriffen erkennung durchzuführen
                                                                                                                 #+----

                yOnes = []
                for j in range(len(climbersThisPic)):
                    #print('\t{}'.format(np.array(climbersThisPic[j][1:2])))
                    yOnes.append(climbersThisPic[j][1:2])
                if yOnes:
                    allClimbers.append(climbersThisPic[np.argmin(yOnes)])                    # die Person mit der BB mit dem geringeren y1 wert, also die die weiter oben ist,
                                                                                            # wird als Kletterer in allClimbers gespeichert
                else:
                    allClimbers.append([0,0,10e-12,10e-12])


                climbersThisPic = []

                # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                # cv2.imwrite(FLAGS.numberedSource + str(count) + '.jpg', img)
                img = draw_persons(img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite(FLAGS.output + str(count) + '.jpg', img)
                logging.info('output saved to: {}'.format(FLAGS.output) + str(count))

            f = open("climbers.txt", "w")
            f.write(str(allClimbers))
            with open(FLAGS.CSVpath + 'time.csv', 'w') as f:
                # create the csv writer
                t1 = time.time()
                writer = csv.writer(f)
                t2 = time.time()
                # write a row to the csv file
                writer.writerow('time: {}'.format(t2 - t1))

#------------------------------------------------------------------------------------------ momentan nicht in verwendung
        elif FLAGS.video is True:                                   # für Detection am cam feed oder uaf dem Video allen
            frameWidth = 1080
            frameHeight = 1920
            if FLAGS.cam is True:
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(FLAGS.vidFile)
            cap.set(3, frameWidth)
            cap.set(4, frameHeight)
            #cap.set(10, 150)
            while True:
                success, img = cap.read()
                imgResult = img.copy()
                imgResult = tf.expand_dims(img_raw, 0)
                imgResult = transform_images(img, FLAGS.size)

                t1 = time.time()
                boxes, scores, classes, nums = yolo(imgResult)
                t2 = time.time()
                logging.info('time: {}'.format(t2 - t1))

                logging.info('detections:')
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                       np.array(scores[0][i]),
                                                       np.array(boxes[0][i])))
                    allhands.append(np.array(boxes[0][i]))  # saving all detected hands in allhands

                img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imshow("Video", imgResult)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        else:
            img_raw = tf.image.decode_image(
                open(FLAGS.image, 'rb').read(), channels=3)
            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                   np.array(scores[0][i]),
                                                   np.array(boxes[0][i])))

            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite(FLAGS.output, img)
            logging.info('output saved to: {}'.format(FLAGS.output))
        # einzel bild detection
#-----------------------------------------------------------------------------------------------------------------------
    # using the hand detection to create a list of all detected Hands


    # 13.08.2021 diese schleife modifizieren um in der richtigen reihenfolge über griffe und bilder zu loopen
    # -> auf überschneidung von griffen und personen boundingboxes achten -> nur dann differenzen berechnen
    #print(str(allClimbers))

    olh = []  # overlapping hands
    points = 0
    holdList = "Gripped Holds:"
    overlapList = 'Overlaps:'
    if FLAGS.baseline:

        for holdID in range(len(holds)):         # alle hände mit allen griffen verschneiden
            logging.info('loading hold Nr.' + str(holdID))

            for index in range(len(allClimbers)):                                        # -> fläche von hand und überschneidung vergleichen
                climbImg = cv2.imread(FLAGS.numberedSource + 'PhotoNr_' + str(index) + '.jpg')

                climberPix = percToPix(allClimbers[index], cv2.imread(FLAGS.baseline))
                #print(climberPix)
                r = overlapRect(holds[holdID], climberPix)

                delay = index - FLAGS.delay                                                     #Delay hier bearbeiten

                #print(rect_area(holds[holdID]), rect_area(r), rect_area(climberPix))



                if index == 0:
                    base = climbImg.copy()
                elif index > FLAGS.delay:                                                       #hier auch Delay bearbeiten
                    base = cv2.imread(FLAGS.numberedSource + 'PhotoNr_' + str(delay) + '.jpg')

                if abs(rect_area(holds[holdID]) - rect_area(r)) < 10e-12 and (index % FLAGS.frameReduction) == 0: # or (index % 600) == 0:
                    olh.append(overlapRect(holds[holdID], climberPix))
                    # print(climbImg)
                    if FLAGS.colorMask is True:                        #WIEDER EINKOMMENTIEREN
                        #img = cv2.imread(str(climbImg))
                        base = mask_colour(base, color)
                        climbImg = mask_colour(climbImg, color)

                    allDiff, score, scorePix = compare_baseline(base, climbImg, holds[holdID])
                    logging.info('overlap detected: image ' + str(index) + ' and hold ' + str(holdID) + '; image similarity ' + str(score) + '; Overlap percent '
                                 + str((allDiff.shape[1] * allDiff.shape[0]-scorePix)/cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
                    overlapList = overlapList + '\n overlap detected: image ' + str(index) + 'and hold' + str(holdID) + '; image similarity ' + str(score)
                    cv2.imwrite(FLAGS.holdsOut + str(holdID) + '/overlapping_' + str(index) + '.jpg', allDiff)
                    if score < FLAGS.similarity: #or ((allDiff.shape[1] * allDiff.shape[0]-scorePix)/cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))) > FLAGS.holdOverlap \
                            #or (allDiff.shape[1] * allDiff.shape[0]-scorePix) > FLAGS.handPix !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! test ohne komische conversion
                        holdList = holdList + " \n hold" #(allDiff.shape[1] * allDiff.shape[0]-scorePix)/cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))
                        print('Size of the hold rectangle: ' + str(allDiff.shape[1] * allDiff.shape[0]))
                        print('Fremdpixel: ' + str(allDiff.shape[1] * allDiff.shape[0] - scorePix))
                        print('NonZero Pixel: ' + str(cv2.countNonZero(cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
                        print('Similarity: ' + str(score))
                        print('Overlap: '+str((allDiff.shape[1] * allDiff.shape[0] - scorePix) / cv2.countNonZero(
                            cv2.cvtColor(allDiff, cv2.COLOR_BGR2GRAY))))
                        cv2.imwrite(FLAGS.holdsOut + str(holdID) + '/gripped_' + str(index) + '.jpg', allDiff)

                        if points < holds[holdID][4]:
                            points = holds[holdID][4]
                        logging.info('progress detected: points = ' + str(points))
                        break
                else:
                    #print(allClimbers[index])

                    logging.info('CLimber Nr. ' + str(index) + ' and Grip Nr. ' + str(holdID) +'no overlap')
        #holdID += 1
    print("Total points: " + str(points))
    # print(holdList)
    # print(overlapList)
    print('Number of detected climbers:' + str(len(allClimbers)))
    # ...
    end = timer()
    #print(end - start)

    logging.info('Elapsed time: {}'.format(str(datetime.timedelta(seconds=(end - start)))))

    #print(np.array(allClimbers))


#----------------------for testing purposes




    #holds = []
    #holds = hold_marker(base)
    #print(holds)



#------------------------------------------------------------------------------------------------------------------#

# Ganz wichtig wäre es vor allem für ein video die überprüfung ob eine überlappung stattfindet in die Detection direkt
# einzubauen.












if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


