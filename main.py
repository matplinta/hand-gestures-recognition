
#! /usr/bin/env python3

import copy
import cv2
import os, sys
import numpy as np
import time
import tensorflow as tf
from utils.Volume import Volume
from utils.detector_utils import WebcamVideoStream
from utils import detector_utils as detector_utils
from multiprocessing import Queue, Pool, Manager, Value
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Settings
prediction = ''
action = ''
score = 0
score_thresh = 0.2
volume = Volume(0)

# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

def predict_rgb_image(img):
    result = gesture_names[model.predict_classes(img)[0]]
    return (result)

def predict_rgb_image_vgg(image, model):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = gesture_names[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def showOriginal(text, frame):
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow(text, frame)

def showPrediction(thresh, prediction, score, action):
    cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))
    cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255))  # Draw the text
    cv2.imshow('Prediction', thresh)

def showContours(thresh):
    threshCopy = copy.deepcopy(thresh)
    contours, hierarchy = cv2.findContours(threshCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
    cv2.imshow('Contours', drawing)

def isHandDetected(detection_graph, sess, frame):
    boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)
    return True if scores[0] > score_thresh else False

def worker(input_q, output_q, bgModel):
    model = load_model('models/VGG_cross_validated.h5')
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        frame = input_q.get()
        if (frame is not None):
            colorImg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = remove_background(frame)
            # img = img[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] 
            if(isHandDetected(detection_graph, sess, colorImg)):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
                ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                target = np.stack((thresh,) * 3, axis=-1)
                target = cv2.resize(target, (224, 224))
                target = target.reshape(1, 224, 224, 3)
                prediction, score = predict_rgb_image_vgg(target, model)
                # print(prediction, score)
                output_q.put([prediction, score])
            else:
                print("Hand not detected")
                output_q.put(['-', 0])
        else:
            print("Frame not received")
            output_q.put(['-', 0])
    sess.close()


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        showOriginal("Press b to set background", cv2.flip(frame, 1))
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # press ESC to exit all windows at any time
            exit()
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            cv2.destroyAllWindows
            camera.release()
            print('Background captured')
            break

    input_q = Queue(maxsize=128)
    output_q = Queue(maxsize=128)

    camera = WebcamVideoStream(src=0, width=640, height=480).start()
    # Camera
    pool = Pool(4, worker, (input_q, output_q, bgModel))
    # while camera.isOpened():
    while True:
        frame = camera.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        frame2 = copy.deepcopy(frame)
        
        input_q.put(frame[0:int(cap_region_y_end * frame.shape[0]), int(cap_region_x_begin * frame.shape[1]):frame.shape[1]])
        output = output_q.get()
        prediction = output[0]
        score = output[1]

        showOriginal("Camera", frame2)
        
        img = remove_background(frame2)
        img = img[0:int(cap_region_y_end * frame2.shape[0]),
                int(cap_region_x_begin * frame2.shape[1]):frame2.shape[1]]  # clip the ROI
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        showPrediction(thresh, prediction, score, action) # draw prediction
        # showContours(thresh) # draw the contours

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if score > 95:
            if prediction == 'Palm':
                action = ""
                # volume.mute()

            elif prediction == 'Fist':
                action = 'Mute'
                volume.mute()

            elif prediction == 'L':
                action = 'Volume up'
                volume.increase(5)

            elif prediction == 'Okay':
                action = 'Volume down'
                volume.decrease(5)

            elif prediction == 'Peace':
                action = 'Volume up'
                volume.increase(5)

            else:
                pass

    pool.terminate() 
    camera.stop()
    cv2.destroyAllWindows()
