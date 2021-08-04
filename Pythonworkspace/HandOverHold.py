import time
from datetime import datetime
from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path
from skimage.metrics import structural_similarity
import numpy as np
import cv2
import tensorflow as tf
import logging
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

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('baseClasses', './data/coco.names', 'path to classes file')
flags.DEFINE_string('baseWeights', './checkpoints/yolov3.tf',
                    'path to weights file')

flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')                       # noch nicht damit experimentiert
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('imDir', './data/', 'path to input image directory')    # Video taken from the same Perspective as Baseline Image (here a directory with just the frames is used -> in a later iteration a direct video feed should be used
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')

flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('baseLine', '', 'Baseline Image of Wall without the climber')  # Baseline Image of Wall without the climber
flags.DEFINE_boolean('detection', False, 'detection Yes or No')


def main(_argv):
    allhands = []

    if baseline:
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
    # using a seperate detector for holds on the bare wall image to set a baseline and saving the results

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

        if FLAGS.tfrecord:
            dataset = load_tfrecord_dataset(
                FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
            dataset = dataset.shuffle(512)
            img_raw, _label = next(iter(dataset.take(1)))
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
                allhands.append(boxes[0][i])                                 # saving all detected hands in allhands

            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite(FLAGS.output, img)
            logging.info('output saved to: {}'.format(FLAGS.output))

        elif FLAGS.imDir:                                                   # für Detection auf allen bildern in imDir
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
                    print(np.array(boxes[0][i]))
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                       np.array(scores[0][i]),
                                                       np.array(boxes[0][i])))

                img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite(FLAGS.output+str(count)+'.jpg', img)
                logging.info('output saved to: {}'.format(FLAGS.output) + str(count))
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
    # using the hand detection on the imDir images

    olh = []                                                         # overlapping hands

    for hold in baseBoxes:                                           # alle hände mit allen griffen verschneiden
        for hand in allhands:                                        # -> fläche von hand und überschneidung vergleichen
            r = overlapRect(hold, hand)
            if rect_area(r) >= 0.75 * rect_area(hand):
                drawRect(r, base, (255, 0, 255), 2)
                olh.append(overlapRect(hold, hand))














if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


