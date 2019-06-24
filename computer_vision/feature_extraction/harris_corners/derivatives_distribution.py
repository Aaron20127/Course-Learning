"""
harris 角点检测的sobel梯度Ix, Iy的分布图。
因为角点检测时的窗口较小，所以图片较小得到的分布效果越好。
三种不同的角点图分别为, corner, edge, flat。

学习PPT：
http://www.cse.psu.edu/~rtc12/CSE486/lecture06.pdf

"""


import cv2 as cv
import os
import sys
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
common_path = abspath + "/../../../common"
data_path = common_path + "/data"
# add the library file path
sys.path.append(common_path)
# change the work path
os.chdir(abspath)

import misc_utils
import math_utils
import opencv

def sobelDerivatives(img_gray, ksize=3):
    """ 获取灰度图的x和y方向的Sobel梯度值
        img_gray: 灰度图，narray，shape(n,m)
        ksize: 梯度算子大小，int
        return: x和y方向的梯度, array，shape(n,)
    """
    row, col = img_gray.shape

    dx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=ksize)
    dy = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=ksize)

    return dx.flatten(), dy.flatten()
 

def derivativesDistribution():
    """ 显示不同的角点类型的x和y方向的梯度分布，显然
        flat分布为小圆形（聚集到原心），edge分布为椭圆形（向某个轴分布），
        corner分布为大圆（向两个轴分布）
        return: None
    """

    filename = ['flat.png', 'edge.png', 'corner.png']

    for img_name in filename:
        img = cv.imread( img_name )
        gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
        gray = np.float32( gray )

        # calculate derivatives
        x, y = sobelDerivatives(gray)

        # show
        misc_utils.plot().plot_base(
                  [x],
                  [y],
                  line_lable = [img_name],
                  line_color = ['#4876FF'],
                  title = img_name,
                  x_lable = 'Ix',
                  y_lable = 'Iy',
                  p_type = ['scatter'],
                  moveAxisToZero = True) 
        
        plt.figure()
        plt.imshow(gray, cmap='gray')
        plt.title(img_name)
        
    plt.show()

if __name__ == "__main__":
    derivativesDistribution()

