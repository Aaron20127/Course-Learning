import cv2 as cv
import sys
import os
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

def zeroCross(src, threshold):
    row, col = src.shape
    img_zero_cross = np.zeros((row, col), dtype = 'uint8')

    for x in range(1, row-1):
        for y in range(1, col-1):
            flag = 0
            
            # 左右
            near_1 = src[x-1, y]
            near_2 = src[x+1, y]
            if (near_1 * near_2 < 0) and \
                (np.abs(near_1 - near_2) > threshold):
                flag += 1

            # 上下
            near_1 = src[x, y-1]
            near_2 = src[x, y+1]
            if (near_1 * near_2 < 0) and \
                (np.abs(near_1 - near_2) > threshold):
                flag += 1

            # 斜对角
            near_1 = src[x+1, y+1]
            near_2 = src[x-1, y-1]
            if (near_1 * near_2 < 0) and \
                (np.abs(near_1 - near_2) > threshold):
                flag += 1

            # 斜对角
            near_1 = src[x+1, y-1]
            near_2 = src[x-1, y+1]
            if (near_1 * near_2 < 0) and \
                (np.abs(near_1 - near_2) > threshold):
                flag += 1

            if flag >= 2:
                img_zero_cross[x,y] = 255

    return img_zero_cross

def LoG():
    """描述：使用非锐化遮蔽和高提升滤波锐化图像，g = f -  k*(f - f')
            g是最后生成的图像，f是原图像，f'是模糊后的图像。
            k类似锐化的比例，K=1为非锐化遮蔽，K>1为高提升滤波，
            K过大过小效果都不好。
       参数: None
       返回: None
         注：感觉这种方法不如Laplacian，不知道用处大不。
    """
    img_original = cv.imread('pictures/plane.bmp', 0)

    # 创建按钮
    button_units = [['threshold', 10, 500],
                    ['sigma', 2, 10],
                    ['n', 25, 30]]

    track_bar = opencv.createTrackbar(button_units, "trackbar")

    while(1):
        cv.waitKey(10)

        # 获取按钮参数
        [threshold, sigma, n] = track_bar.getTrackbarPos()
        
        n = 6*sigma + 1
        img_gaussian = cv.GaussianBlur(img_original, (n, n), sigmaX=sigma, sigmaY=sigma)


        # img_LOG = cv.Laplacian(img_gaussian, cv.CV_16S, ksize = 3)
        lapla_filter = np.array([[ -1, -1, -1],
                                 [ -1,  8, -1],
                                 [ -1, -1, -1]])

        img_LOG = cv.filter2D(img_gaussian, cv.CV_64F, lapla_filter)

        ## 2.零交叉检测
        img_zero_cross = zeroCross(img_LOG, threshold)

        # 显示灰度图
        # cv.namedWindow('img_original', 0)
        # cv.namedWindow('img_gaussian', 0)
        # cv.namedWindow('img_LOG', 0)
        # cv.namedWindow('img_zero_cross', 0)

        # cv.imshow('img_original', img_original)
        # cv.imshow('img_gaussian', img_gaussian)
        # cv.imshow('img_LOG', opencv.normlizeImage2Uint8(img_LOG))
        cv.imshow('img_zero_cross', img_zero_cross)


if __name__=="__main__":
    LoG()