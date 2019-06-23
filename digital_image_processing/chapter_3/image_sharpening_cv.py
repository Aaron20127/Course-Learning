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

"""
使用sobel算子对图像x和y方向求微分，微分会导致求出的微分值有正有负，
所以使用plt显示的时候最好把vmin和vmax设置成微分图像最小和最大值。

问题：函数中sobel同时对x和y求梯度
"""

img = cv.imread('pictures/box.png',0)

sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

# 1,1方向的sobel是啥？
sobelxy = cv.Sobel(img,cv.CV_64F,1,1,ksize=5)
sobelxy_add = sobelx + sobely

# 两种取梯度值的方法
sobelxy_abs = np.abs(sobelx) + np.abs(sobely)
# sobelxy_sqrt = np.sqrt(sobelx**2 + sobely**2)

plt.subplot(2,3,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(sobelx, cmap = 'gray')
plt.title('sobelx'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobely, cmap = 'gray')
plt.title('sobely'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobelxy, cmap = 'gray')
plt.title('sobelxy'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(sobelxy_add, cmap = 'gray')
plt.title('sobelxy_add'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(sobelxy_abs, cmap = 'gray')
plt.title('sobelxy_abs'), plt.xticks([]), plt.yticks([])

plt.show()
