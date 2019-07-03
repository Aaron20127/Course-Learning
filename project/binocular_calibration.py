import os
import cv2 as cv
import sys
import numpy as np
import copy
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../common")
# change the work path
os.chdir(abspath)

import misc_utils
import opencv
import glob



images = []
# imagepath = glob.glob(abspath + '/calibration/img_2_10.png')
imagepath = glob.glob(abspath + '/binocular/camera1/*.png')
for fname in imagepath:
    img = cv.imread(fname)
    images.append(img)

ret, mtx, dist, rvecs, tvecs = opencv.cameraCalibration(images, 9, 6, showFindCorner=True)


