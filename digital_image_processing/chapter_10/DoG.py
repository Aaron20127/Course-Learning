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

def zeroCross(src, threshold):
    """描述：使用3*3的格子判断一个点是否是边缘，中心点的上下、左右、左上右下、右上左下这四对
            点，至少有两对点符号相反，且这两对点差值的绝对值大于某个给定的阈值。则视为边缘点，保留。
    """
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


def DoG():
    """描述：LoG的近似方法，但是计算速度要比LoG快很多因为只有一个卷积。
             (G1(x,y) - G2(x,y)) * f(x,y)
       参数: None
       返回: None
       注：由于没有找到高斯核，所以这里的实现将减号展开了，相当于进行了两次卷积
    """
    img_original = cv.imread('pictures/building.tif', 0)

    # 创建按钮
    button_units = [['threshold', 3, 100],
                    ['sigma', 4, 10],
                    ['n', 25, 30]]

    track_bar = opencv.createTrackbar(button_units, "trackbar")

    while(1):
        cv.waitKey(10)

        # 获取按钮参数
        [threshold, sigma, n] = track_bar.getTrackbarPos()
        
        ## 1.DoG, 高斯滤波相减 sigma = 4, sigma1 = 5.15, sigma2 = 3.22，
        #  根据关系式10.2-27(刚撒雷斯教材), 且sigma1 = 1.6 * sigma2
        sigma1 = 5.15
        n1 = int(sigma1 * 3) * 2 + 3
        img_gaussian_1 = cv.GaussianBlur(img_original, (n1, n1), sigmaX=sigma1, sigmaY=sigma1)

        sigma2 = 3.22
        n2 = int(sigma2 * 3) * 2 + 3
        img_gaussian_2 = cv.GaussianBlur(img_original, (n2, n2), sigmaX=sigma2, sigmaY=sigma2)

        img_DOG = np.float32(img_gaussian_1) - np.float32(img_gaussian_2)

        ## 2.零交叉检测
        img_zero_cross = zeroCross(img_DOG, threshold)
   

        # 显示灰度图
        # cv.namedWindow('img_original', 0)
        # cv.namedWindow('img_gaussian_1', 0)
        # cv.namedWindow('img_gaussian_2', 0)
        # cv.namedWindow('img_DOG', 0)
        cv.namedWindow('img_zero_cross', 0)
        # cv.imshow('img_original', img_original)
        # cv.imshow('img_gaussian_1', img_gaussian_1)
        # cv.imshow('img_gaussian_2', img_gaussian_2)
        # cv.imshow('img_DOG', opencv.normlizeImage2Uint8(img_DOG))
        cv.imshow('img_zero_cross', img_zero_cross)


if __name__=="__main__":
    DoG()