from numpy import *
from scipy import * 

import random
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


def noiseGauss(src, mu, sigma):
    """ 描述: 灰度图的每个像素随机产生高斯噪声
        参数: src: 灰度矩阵, narray, shape(m,n)
              mu：噪声均值
              sigma: 噪声方差
        返回：带噪声的灰度图, narray, shape(m,r)
        疑问：
              1.高斯噪声是每个像素都要添加噪声吗？
              2.当噪声大于255或小于0时是否需要截断？
    """
    img = copy.deepcopy(src)
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            # 每个点随机的产生高斯噪声
            if random.randint(0,1)==0:
                noise = random.gauss(mu, sigma)
                new = img[i][j].astype("float64") + noise
                if new > 255:
                    new = 255
                elif new < 0:
                    new = 0
                img[i][j] = np.uint8(new)
    return img


def noiseSaltAndPepper(src, rate, salt=255, pepper=0):  
    """ 描述: 随机产生椒盐噪声,椒和盐是均匀分布的，出现的概率各自占50%
        参数: src: 灰度矩阵, narray, shape(m,n)
              percetage: 可能有多少个像素产生噪声
        返回：带噪声的灰度图, narray, shape(m,r)
        疑问：
             1.每次产生椒盐噪声的位置是用的是均匀分布吗？
             2.每次产生的噪声是叠加到像素上还是直接替换像素？
             3.替换像素的值是可以是0-255的任意值？
             4.椒盐必须是0和255像素吗？
    """
    img = copy.deepcopy(src)

    row, col = img.shape
    for i in range(row):
        for j in range(col):
            # 椒盐各为0.25概率
            num = random.randint(0,4)
            if num==0: 
                img[i,j]=salt 
            elif num==1: 
                img[i,j]=pepper 
    return img


def gometricMeanFiltering(src, size=3):
    """ 描述: 几何均值滤波，比如3核的滤波器，为核中所有像素的乘积，再开9次方。
              f(x,y) = (a11 * a12 * ... a33)**(1/9)。
        参数: src: 带噪声的灰度图, narray, shape(m,n)
              size: 核的大小, 为奇数3,7..
        返回：滤波后图片, narray, shape(m,n)
    """
    img = copy.deepcopy(src).astype("float64")
    border = int((size-1) / 2)
    # 卷积之前先在原图扩充一层边界
    img_border = cv.copyMakeBorder\
        (img,border,border,border,border,cv.BORDER_DEFAULT)

    row, col = img.shape

    for i in range(row):
        for j in range(col):
            slide = img_border[i:i+size,j:j+size]
            img[i,j] = np.prod(slide)**(1.0/(size*size))
            # 越界的像素被截取
            if img[i,j] < 0:
                img[i,j] = 0
            elif img[i,j] > 255:
                img[i,j] = 255
    
    return img.astype("uint8")


def medianFiltering(src, size=3):
    """ 描述: 中值滤波，每次选择滤波核中的中间值
        参数: src: 带噪声的灰度图, narray, shape(m,n)
              size: 核的大小, 为奇数3,7..
        返回：滤波后图片, narray, shape(m,n)
    """
    img = copy.deepcopy(src)
    border = int((size-1) / 2)
    # 卷积之前先在原图扩充一层边界
    img_border = cv.copyMakeBorder\
        (img,border,border,border,border,cv.BORDER_DEFAULT)

    row, col = img.shape

    for i in range(row):
        for j in range(col):
            slide = img_border[i:i+size,j:j+size]
            # 因为slide是奇数，np.median所以每次取中值是中间的数
            img[i,j] = np.median(slide)
    
    return img


def adaptiveMedianFiltering(src, size=3, max_size=7):
    """ 描述: 自适应中值滤波，从小的核到大的核中值滤波。
              核越小，细节保留越完整；
              核越大越容易滤除椒盐噪声，但是图像会越模糊；
        参数: src: 带噪声的灰度图, narray, shape(m,n)
              size: 初始核的大小, 为奇数3,7..
              max_size: 最大滤波核
        返回：滤波后图片, narray, shape(m,n)
    """
    img = copy.deepcopy(src.astype("float64"))

    # 拓展不同的和使用的边缘
    img_borders = {}
    for i in range(int((max_size - size)/2)+1):
        img_border = cv.copyMakeBorder(\
            img,i+1,i+1,i+1,i+1,cv.BORDER_DEFAULT) 
        img_borders[size + 2*i] = img_border


    row, col = img.shape

    # 自适应中值滤波算法
    for i in range(row):
        for j in range(col):
            size_cur = size
            while(1):
                img_border = img_borders[size_cur]
                slide = img_border[i:i+size_cur,j:j+size_cur]
                xy = int((size_cur-size)/2)+1
                v_xy  = slide[xy,xy]
                v_min = slide.min()
                v_max = slide.max()
                v_med = np.median(slide)

                if (v_med - v_min > 0) and (v_med - v_max < 0): 
                    if (v_xy - v_min > 0) and (v_xy - v_max < 0):
                        img[i,j] = v_xy
                        break
                    else:
                        img[i,j] = v_med
                        break
                else:
                    if (size_cur >= max_size):
                        img[i,j] = v_med
                        break

                size_cur += 2

    return img.astype("uint8")



def main():

    # img = cv.imread('pictures/pattern.tif',0)
    # img = cv.imread('pictures/ckt-board.tif',0)
    img = cv.imread('pictures/lena.bmp',0)

    # cv.imshow('img', img)
    # cv.imshow('img_hist', opencv.drawHist(img))

    # 创建按钮
    button_units = [['gauss_sigma', 20, 100],
                    ['salt_rate', 40, 100],
                    ['salt', 255, 255],
                    ['pepper', 0, 255]]

    trackbar = opencv.createTrackbar(button_units, "trackbar")

    while(1):
        cv.waitKey(1)

        # 获取按钮参数
        [gauss_sigma, salt_rate, salt, pepper] = trackbar.getTrackbarPos()
        salt_rate = salt_rate / 100.0

        # 添加噪声
        img_gaussNoise = noiseGauss(img, 0, gauss_sigma)
        img_saltNoise = noiseSaltAndPepper(img, salt_rate, salt, pepper)
        cv.imshow('img_gaussNoise', img_gaussNoise)
        cv.imshow('img_saltNoise', img_saltNoise)
        # cv.imshow('img_gaussNoise_hist', opencv.drawHist(img_gaussNoise))
        # cv.imshow('img_saltNoise_hist', opencv.drawHist(img_saltNoise))


        # 几何均值滤波去燥
        gometricMeanFiltering_gaussNoise = gometricMeanFiltering(img_gaussNoise)
        gometricMeanFiltering_saltNoise = gometricMeanFiltering(img_saltNoise)
        cv.imshow('gometricMeanFiltering_gaussNoise', gometricMeanFiltering_gaussNoise)
        cv.imshow('gometricMeanFiltering_saltNoise', gometricMeanFiltering_saltNoise)


        # 中值滤波
        medianFiltering_gaussNoise = medianFiltering(img_gaussNoise,size=3)
        medianFiltering_saltNoise = medianFiltering(img_saltNoise,size=3)
        cv.imshow('medianFiltering_gaussNoise', medianFiltering_gaussNoise)
        cv.imshow('medianFiltering_saltNoise', medianFiltering_saltNoise)


        # 自适应中值滤波
        adaptiveMedianFiltering_gaussNoise = adaptiveMedianFiltering(img_gaussNoise)
        adaptiveMedianFiltering_saltNoise  = adaptiveMedianFiltering(img_saltNoise)
        cv.imshow('adaptiveMedianFiltering_gaussNoise', adaptiveMedianFiltering_gaussNoise)
        cv.imshow('adaptiveMedianFiltering_saltNoise', adaptiveMedianFiltering_saltNoise)


if __name__=="__main__":
    main()
