import os
import cv2
import numpy as np
import glob
import t1 
from matplotlib import pyplot as plt


DIGITS_LOOKUP = [
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1]
]

DIGITS = [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
]

DIGIT_TEST = [
    [0, 1, 1, 0, 1, 1, 1],
]


def train():

    global model

    # samples = np.array(DIGITS_LOOKUP,np.float32)
    # responses = np.array(DIGITS,np.float32)

    samples,responses = t1.readImages()
    print("Samples ",np.shape(samples)," item: ",np.shape(samples[0]))
    print("Responses ",np.shape(responses))

    model = cv2.ml.KNearest_create()
    model.train(samples,cv2.ml.ROW_SAMPLE,responses)

def decode(img):

    global model
    
    # test = np.array(DIGIT_TEST,np.float32)
    # test = t1.loadImage("06755299_0_6.png")
    # test = t1.loadImage("44187543_5_3.png")

    test = t1.readImage(img,isdebug=False)
    test = np.array([test],np.float32)

    print("Test Len: ",np.shape(test))
    #print("Test Image: ",test)

    ret, results, neighbours ,dist = model.findNearest(test, 3)
    print( "result:  {}\n".format(results) )
    print( "neighbours:  {}\n".format(neighbours) )
    print( "distance:  {}\n".format(dist) )

    result = int(results[0])
    print(" Result: ",result)

    return result


