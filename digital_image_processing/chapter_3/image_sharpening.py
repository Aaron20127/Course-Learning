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



def laplacianGradient():
    """描述：使用laplacian算子使图像锐化
       参数: None
       返回: None
    """
    img_gray = cv.imread('pictures/moon.tif', 0)

    # 微分算子核
    laplacian_derivatives_8 = np.array([[ 1, 1, 1],
                                        [ 1,-8, 1],
                                        [ 1, 1, 1]])

    # 叠加在原图像上的微分算子核，具有45度旋转不变性
    laplacian_sharpening_8 = np.array([[ -1, -1, -1],
                                       [ -1,  9, -1],
                                       [ -1, -1, -1]])
    
    # 叠加在原图像上的微分算子核，具有90度旋转不变性
    laplacian_sharpening_4 = np.array([[  0, -1,  0],
                                       [ -1,  5, -1],
                                       [  0, -1,  0]])

    # 边界滤波的时候，先使用 BORDER_DEFAULT 镜像插值，扩大原图像的边界
    img_derivatives_8 = cv.filter2D(img_gray, -1, laplacian_derivatives_8)
    img_sharpening_8 = cv.filter2D(img_gray, -1, laplacian_sharpening_8)
    img_sharpening_4 = cv.filter2D(img_gray, -1, laplacian_sharpening_4)

     #     cv.imshow("img_gray", img_gray)
     #     cv.imshow("img_derivatives_8", img_derivatives_8)
     #     cv.imshow("img_sharpening_8", img_sharpening_8)
     #     cv.imshow("img_sharpening_4", img_sharpening_4)
     #     cv.waitKey(0)
    plt.figure(),plt.imshow(img_gray,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.figure(),plt.imshow(img_derivatives_8,cmap = 'gray')
    plt.title('img_derivatives_8'), plt.xticks([]), plt.yticks([])

    plt.figure(),plt.imshow(img_sharpening_8,cmap = 'gray')
    plt.title('img_sharpening_8'), plt.xticks([]), plt.yticks([])

    plt.figure(),plt.imshow(img_sharpening_4,cmap = 'gray')
    plt.title('img_sharpening_4'), plt.xticks([]), plt.yticks([])

    plt.show()
    


def unsharpMaskingAndHighboostFiltering():
    """描述：使用非锐化遮蔽和高提升滤波锐化图像，g = f -  k*(f - f')
            g是最后生成的图像，f是原图像，f'是模糊后的图像。
            k类似锐化的比例，K=1为非锐化遮蔽，K>1为高提升滤波，
            K过大过小效果都不好。
       参数: None
       返回: None
         注：感觉这种方法不如Laplacian，不知道用处大不。
    """
    img_original = cv.imread('pictures/moon.tif', 0)

    # 创建按钮
    button_units = [['sigma', 3, 10],
                    ['k', 10, 100]]

    track_bar = opencv.createTrackbar(button_units, "trackbar")

    while(1):
        cv.waitKey(1)

        # 获取按钮参数
        [sigma, k] = track_bar.getTrackbarPos()
        k = k / 20

        
        img_gaussian = cv.GaussianBlur(img_original, (5, 5), sigmaX=sigma, sigmaY=sigma)

        # 使用float64不会产生截取，因此不会损失数据
        unshape_mask = img_original.astype("float64") - img_gaussian.astype("float64")
        img_unsharp_mask = img_original.astype("float64") + unshape_mask
        # add将相加后的数据变成uint8型，且小于0的数截断变成0，大于255的数截断，变成255
        img_highboost_Filter = cv.add(img_original.astype("float64"), k * unshape_mask, dtype=cv.CV_8U)

        # 显示灰度图
        cv.imshow('img_original', img_original)
        cv.imshow('img_gaussian', img_gaussian)
        cv.imshow('unshape_mask', opencv.normlizeImage2Uint8(unshape_mask))
        cv.imshow('img_unsharp_mask', opencv.normlizeImage2Uint8(img_unsharp_mask))
        cv.imshow('img_highboost_Filter', img_highboost_Filter)


def sobelGradient():
    """描述：使用sobel算子使图像锐化,可以发现sobel算子得到的边缘更粗一些，
             而laplacian得到的边缘更细一些。因为laplacian是二阶微分，
             sobel是一阶微分。
       参数: None
       返回: None
    """
    img_original = cv.imread('pictures/box.png', 0)

    # laplacian算子
    laplacian_derivative = np.array([[ -1, -1, -1],
                                     [ -1,  8, -1],
                                     [ -1, -1, -1]])

    # sobel算子核, 因为最后的梯度值是x与y方向的绝对值之和，所以
    # sobel求梯度的两个核要分开算
    sobel_derivative_x_filter = np.array([[ -1, -2, -1],
                                          [  0,  0, 0],
                                          [  1,  2, 1]])

    sobel_derivative_y_filter = np.array([[ -1,  0, 1],
                                          [ -2,  0, 2],
                                          [ -1,  0, 1]])


    # 边界滤波的时候，先使用 BORDER_DEFAULT 镜像插值，扩大原图像的边界
    # laplacian是二阶微分
    img_laplacian_derivative = cv.filter2D(img_original, -1, laplacian_derivative)

    # sobel是一阶微分，在box中部分微分值是负数，部分微分值是正数
    img_sobel_derivatives_x = cv.filter2D(img_original, cv.CV_64F, sobel_derivative_x_filter)
    img_sobel_derivatives_y = cv.filter2D(img_original, cv.CV_64F, sobel_derivative_y_filter)
    # 将微分值取绝对值，使负微分值变成正的微分值
    img_sobel_derivatives = cv.addWeighted(\
        np.abs(img_sobel_derivatives_x), 1, np.abs(img_sobel_derivatives_y), 1, 0)


    misc_utils.plot().plot_multiple_picture((1,1),\
         [[img_original, 'img_original', 'gray']])
    misc_utils.plot().plot_multiple_picture((1,1),\
         [[img_laplacian_derivative, 'img_laplacian_derivative', 'gray']])
    misc_utils.plot().plot_multiple_picture((1,1),\
         [[img_sobel_derivatives_x, 'img_sobel_derivatives_x', 'gray']])
    misc_utils.plot().plot_multiple_picture((1,1),\
         [[img_sobel_derivatives_y, 'img_sobel_derivatives_y', 'gray']])
    misc_utils.plot().plot_multiple_picture((1,1),\
         [[img_sobel_derivatives, 'img_sobel_derivatives', 'gray']])

    plt.show()

if __name__=="__main__":
    # laplacianGradient()
    # unsharpMaskingAndHighboostFiltering()
    sobelGradient()
