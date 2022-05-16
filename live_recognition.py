# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from keras.engine import Model
from keras_vggface.vggface import VGGFace
## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Softmax, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
from Functions_List import *
import serial, time

def detect_and_predict(frame, faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (416, 416),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.85:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it

            locs.append((startX, startY, endX, endY))

    return locs


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

model = loadModel()
# Load VGG Face model weights
model.load_weights('vgg_face_weights.h5')
# model.load_weights('colab_Best_Weights_VGG16_.h5')

# Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
vgg_face = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Load saved model
classifier_model = tf.keras.models.load_model('lt_face_classifier_model.h5')
person_rep = {0: 'Abdullah',1: 'Adnan' ,2: 'Aleza_shah', 3:'Ayeza_khan',4:'Baber' ,5:'I.Khan' ,6: 'Mark zakr', 7:'Salman',8: 'Tahir' ,9:'Trump'}


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# initialize the video stream
lock_name = ''
i = 0
# loop over the frames from the video stream

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()

    frame = imutils.resize(frame, width=600)
    locs = []

    key = cv2.waitKey(1) & 0xFF
    if key == ord("m"):
        # detect faces in the frame and determine if they are wearing a
        # face mask or not

        locs = detect_and_predict(frame, faceNet)
    persons_in_img = []
    score_in_img = []
    if len(locs) > 0:
        # loop over the detected face locations and their corresponding
        # locations
        for box in locs:
            # unpack the bounding box and predictions

            (startX, startY, endX, endY) = box

            img_crop = frame[startY:endY, startX:endX]
            img_crop = cv2.resize(img_crop, (224, 224))
            res_img = img_crop
            crop_img = img_to_array(img_crop)
            crop_img = np.expand_dims(crop_img, axis=0)
            crop_img = preprocess_input(crop_img)
            img_encode = vgg_face(crop_img)

            # Make Predictions
            embed = K.eval(img_encode)
            person = classifier_model.predict(embed)
            # print(max(person)*100)
            if max(max(person)) > 0.88:
                name = person_rep[np.argmax(person)]
                # # cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                # # img = cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
                # # img = cv2.putText(img, str(np.max(person)), (right, bottom + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                # #                   cv2.LINE_AA)
                persons_in_img.append((max(max(person)), res_img, name, box))
                # score_in_img.append(max(max(person)))
                # # Save images with bounding box,name and accuracy
                #
                # cv2.putText(frame, str(np.max(person)), (startX, startY - 20),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 2)
                # cv2.putText(frame, name, (startX, startY - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
                # cv2.rectangle(frame, (startX, startY), (endX, endY), (255,255,255), 2)

            else:
                name = "Unknown"
                persons_in_img.append((max(max(person)), res_img, name, box))
    # cv2.imwrite('v1Predictions/'+str(i)+'.png', frame)
    # if score_in_img:
    #     print("Brfore sorting: ",score_in_img)
    # Z = [x for _, x in sorted(zip(score_in_img, persons_in_img))]
    # btn=input("want to recgonise? press 1")

    if persons_in_img and key == ord("m"):
        persons_in_img.sort()
        score, img, name, box = persons_in_img[-1]
        (startX, startY, endX, endY) = box

        cv2.putText(frame, str(score), (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0,255,0), 2)
        cv2.putText(frame, name, (startX, startY- 20),cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,0), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 1)
        cv2.imshow("Result", frame)
        cv2.imshow("Frame", frame)
        cv2.imwrite('person_detected.png', img)
        new_sendmail(name)
        store_csv(name)

        lock_name = name
        print("formatting")
        # send_mail(name)


    # print(persons_in_img[:,0])
    i = i + 1
    cv2.imshow("Frame", frame)
    time.sleep(0.1)
    if lock_name != 'Unknown' and lock_name != '':
        lock_open(lock_name)
    lock_name = ''
    locs = []
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
