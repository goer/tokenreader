import os
import cv2
import numpy as np
import glob
import t1 

img_array = t1.loadImage("06755299_0_6.png")
#print(img_array)
#print(len(img_array))

x,y = t1.readImages()
print(x)
print(y)