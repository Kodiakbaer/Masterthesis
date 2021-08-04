import cv2
import numpy as np
YoloPath = 'Resources/YOLO3/'

img = cv2.imread("Resources/TestWalls/TestWall.png")
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classFile = YoloPath + 'coco.names'     # Class names definieren
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
# print(classNames)

modelConfiguration = YoloPath + 'yolov3-320.cfg'
modelWeights = YoloPath + 'yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confval = []

    for output in outputs:
        for det in output:
            scores  = det[5:]
            classId = np.argmax(scores)
            confidence = scores [classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3] * hT)
                x, y = int(det[0] * wT - w/2), int(det[1] * hT - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confval.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confval, confThreshold, nmsThreshold)
    print(indices)
    for i in indices:
        i = i[0]
        box =bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confval[i]*100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)





#while True:

blob =cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
net.setInput(blob)

layerNames = net.getLayerNames()
# print(layerNames)
outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]  # reduce index by 1 because we are starting to count on 0 in python

# print(outputNames)

outputs = net.forward(outputNames)

# print(outputs[0].shape)
# print(outputs[1].shape)
# print(outputs[2].shape)
# print(outputs[0][0])

findObjects(outputs,img)

cv2.imshow('Image', img)
cv2.waitKey(0)
#if cv2.waitKey(1)  & 0xFF == ord('q'):
        #break