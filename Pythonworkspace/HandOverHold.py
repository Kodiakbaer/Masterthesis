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

##################### conditions #####################
flags.DEFINE_boolean('detection', False, 'detection Yes or No')
flags.DEFINE_boolean('holdsDetection', False, 'Hold detection Yes or No')
flags.DEFINE_boolean('cam', False, 'is a cam used?')
flags.DEFINE_boolean('video', False, 'is a video as a file or cam feed used?')

###################### outputs ######################
flags.DEFINE_string('output', './output.jpg', 'path to output image')

#----------------------for testing purposes
basePath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1000.jpg"
#basePath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1921.jpg"
#basePath = "D:\MMichenthaler\VideoFrames\Video2\Video2_frame1871.jpg"
base = cv2.imread(basePath)

baseRect = np.array([584, 1036, 625, 1069])
color = [0, 24, 27, 178, 190, 146]
#aoi = np.array(base[baseRect[1]:baseRect[3], baseRect[0]:baseRect[2]])



#----------------------------------------------------------------------------------------------------------------------#

def main(_argv):

    allClimbers = []
    holds = hold_marker(base)
    print(holds)

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

    elif FLAGS.baseline:
        holds = hold_marker(base)
        print(holds)

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

        if FLAGS.imDir:
            for count, dirImg in enumerate(Path(FLAGS.imDir).iterdir()):
                img_raw = tf.image.decode_image(
                    open(dirImg, 'rb').read(), channels=3)

                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, FLAGS.size)

                t1 = time.time()
                boxes, scores, classes, nums = yolo(img)
                t2 = time.time()
                logging.info('time: {}'.format(t2 - t1))

                logging.info('detections:')

                for i in range(nums[0]):
                    #print(np.array(boxes[0][i]))
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                       np.array(scores[0][i]),
                                                       np.array(boxes[0][i])))
                    if class_names[int(classes[0][i])] == "person" and scores[0][i] > 0.6:
                        allClimbers.append(boxes[0][i])                           # saving all detected hands in allhands

                img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                img = draw_persons(img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite(FLAGS.output+str(count)+'.jpg', img)
                logging.info('output saved to: {}'.format(FLAGS.output) + str(count))

        # für Detection auf allen bildern in imDir
        elif FLAGS.video is True:
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
                    allhands.append(boxes[0][i])  # saving all detected hands in allhands

                img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imshow("Video", imgResult)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        # für Detection am cam feed oder uaf dem Video allen

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
    # using the hand detection to create a list of all detected Hands


    # 13.08.2021 diese schleife modifizieren um in der richtigen reihenfolge über griffe und bilder zu loopen
    # -> auf überschneidung von griffen und personen boundingboxes achten -> nur dann differenzen berechnen
    olh = []                                                         # overlapping hands
    if FLAGS.baseline:
        for hold in holds:                                           # alle hände mit allen griffen verschneiden
            for climber in allClimbers:                                        # -> fläche von hand und überschneidung vergleichen
                r = overlapRect(hold, climber)
                if rect_area(r) >= 0.75 * rect_area(climber):
                    drawRect(r, base, (255, 0, 255), 2)
                    olh.append(overlapRect(hold, climber))
    print(np.array(allClimbers))


#----------------------for testing purposes
    '''   verschneidungstest von mehrern hand rechtecken mit einem griff rechteck 
    allhands = [[0.41006285, 0.42347014, 0.7614191, 0.8621985],
                [0.409926, 0.42104656, 0.7612273, 0.86492175],
                [0.4092157, 0.41622725, 0.76479065, 0.8680327],
                [0.41460437, 0.41787606, 0.7606239, 0.86354953],
                [0.41505966, 0.41920617, 0.76014113, 0.86396134],
                [0.4183203, 0.41473258, 0.75801873, 0.8666041],
                [0.4192586, 0.42035326, 0.75870824, 0.86138284],
                [0.41785425, 0.42000327, 0.7576223, 0.85765004],
                [0.41965738, 0.41426432, 0.7552775, 0.8637074],
                [0.4204809, 0.4154017, 0.7525857, 0.8634217],
                [0.4196638, 0.41141456, 0.7530681, 0.86506754],
                [0.41827935, 0.415114, 0.75403935, 0.86354786],
                [0.41740987, 0.41588917, 0.7541821, 0.8621843],
                [0.4160223, 0.41886348, 0.75506985, 0.86216277],
                [0.41614324, 0.4168234, 0.7540466, 0.8606248],
                [0.41635805, 0.42132342, 0.75407976, 0.85889864],
                [0.41341496, 0.42059684, 0.75582683, 0.8633392],
                [0.415411, 0.41905028, 0.75331414, 0.8632632],
                [0.4156507, 0.4216166, 0.7545028, 0.86404556],
                [0.41548878, 0.4221872, 0.754783, 0.8620622],
                [0.42002618, 0.4221908, 0.7520453, 0.8624232],
                [0.41660357, 0.42034984, 0.7518734, 0.86518073],
                [0.4169307, 0.41948748, 0.75159836, 0.86633277],
                [0.41918078, 0.42060584, 0.7498467, 0.8665597],
                [0.41915, 0.41995704, 0.74960124, 0.86642694],
                [0.41898906, 0.41726893, 0.7476455, 0.86694056],
                [0.41848812, 0.41325915, 0.74770606, 0.87038124]]
    print(base.shape)
    baseRectPercent = pixToPercent(baseRect, base)
    baseArea = rect_area(baseRectPercent)

    
    for i in range(len(allhands)):
        detectArea = rect_area(overlapRect(np.array(allhands[i]), baseRectPercent))

        if detectArea == baseArea:
            compRes, score = compare_baseline(base, cv2.imread(FLAGS.output+str(i)+'.jpg'), baseRect)
            print(score)
            cv2.imwrite(FLAGS.output+str(i)+'baseComparison.jpg', compRes)
        else:
            print("oh no!")
       test = cv2.imread("D:\MMichenthaler\HandOverHold\In\BilddifferenzTestung\Frames\Video2_frame1947.jpg")
    cv2.imshow("base1", base)
    cv2.imshow("test1", test)
    Diff, sc = compare_baseline(base, test)

    cv2.imshow("test", Diff)
    cv2.imshow("base", base)
    cv2.imshow("compare", test)
    cv2.waitKey(0)

    '''


    #holds = []
    #holds = hold_marker(base)
    #print(holds)

    '''Test loop für bilddifferenzen in rechteck
    for count, dirImg in enumerate(Path(FLAGS.imDir).iterdir()):
        print(dirImg)
        if count == 0 or (count % 60) == 0:
            base = cv2.imread(str(dirImg))
        img = cv2.imread(str(dirImg))
        baseMasked = mask_colour(base, color)
        imgMasked =  mask_colour(img, color)
        allDiff, score = compare_baseline(baseMasked, imgMasked, baseRect)

        cv2.imwrite(FLAGS.output + str(count) + '.jpg', allDiff)
        print(count)
        print(score)                #vielleicht anhand von score als gegriffen zählen -> nicht mehr stark verändernd und ähnlichkeit unter gewissem wert
    '''



#------------------------------------------------------------------------------------------------------------------#

# Ganz wichtig wäre es vor allem für ein video die überprüfung ob eine überlappung stattfindet in die Detection direkt
# einzubauen.












if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


