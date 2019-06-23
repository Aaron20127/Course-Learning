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


def grayScaleInversion():
    """描述：反转查看暗色区域的一些细节
    """
    img_BGR = cv.imread('pictures/negative_transformation.jpg')
    img_gray = cv.cvtColor(img_BGR,cv.COLOR_BGR2GRAY)

    # 灰度反转
    img_negative = 255 - img_gray

    data = [[img_gray, 'Original', 'gray'],
            [img_negative, 'L - 1 - r', 'gray']]

    misc_utils.plot().plot_multiple_picture((1,2), data)
    plt.show()


if __name__=="__main__":   
    grayScaleInversion()