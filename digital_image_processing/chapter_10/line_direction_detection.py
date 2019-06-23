import cv2 as cv
import os
import sys
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)

import misc_utils
import math_utils
import opencv


def main():
    # img_original = cv.imread('pictures/wirebond_mask.tif', 0)
    img_original = cv.imread('pictures/building.tif', 0)

    ### 1. 不滤波，直接求导数
    # 不同角度检测子
    degree00 = np.array([[ -1, -2, -1],
                         [  0,  0,  0],
                         [  1,  2,  1]])

    # degree45 = np.array([[ 0,  1,  2],
    #                     [ -1,  0,  1],
    #                     [ -2, -1,  0]])
    
    ## 这个45度，比索贝尔45度效果好
    degree45 = np.array([[ 2,  -1,  -1],
                         [ -1,  2,  -1],
                         [ -1,  -1,  2]])


    degree90 = np.array([[ -1,  0,  1],
                         [ -2,  0,  2],
                         [ -1,  0,  1]])

    img_degree00 = cv.filter2D(img_original, cv.CV_64F, degree00)
    img_degree45 = cv.filter2D(img_original, cv.CV_64F, degree45)
    img_degree90 = cv.filter2D(img_original, cv.CV_64F, degree90)


    # 显示灰度图
    cv.namedWindow('original', 0)
    cv.namedWindow('degree00', 0)
    cv.namedWindow('degree45', 0)
    cv.namedWindow('degree90', 0)

    cv.imshow('original', img_original)
    cv.imshow('degree00', opencv.normlizeImage2Uint8(img_degree00))
    cv.imshow('degree45', opencv.normlizeImage2Uint8(img_degree45))
    cv.imshow('degree90', opencv.normlizeImage2Uint8(img_degree90))


    ### 2.先模糊再求梯度
    # 先滤波，去除非主要边缘
    img_original = cv.GaussianBlur(img_original, (19, 19), sigmaX=3, sigmaY=3)
    
    # 不同角度检测子
    img_degree00 = cv.filter2D(img_original, cv.CV_64F, degree00)
    img_degree45 = cv.filter2D(img_original, cv.CV_64F, degree45)
    img_degree90 = cv.filter2D(img_original, cv.CV_64F, degree90)


    # 显示灰度图
    cv.namedWindow('degree00_filter', 0)
    cv.namedWindow('degree45_filter', 0)
    cv.namedWindow('degree90_filter', 0)

    cv.imshow('degree00_filter', opencv.normlizeImage2Uint8(img_degree00))
    cv.imshow('degree45_filter', opencv.normlizeImage2Uint8(img_degree45))
    cv.imshow('degree90_filter', opencv.normlizeImage2Uint8(img_degree90))

    cv.waitKey(0)


if __name__=="__main__":
    main()