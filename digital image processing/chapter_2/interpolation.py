

import cv2 as cv
import sys
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

sys.path.append("D:/common") 
import misc_utils
import math_utils
import opencv


def interpolation():
    """INTER_NEAREST：最邻近，效果最差
       INTER_AREA：   后边三个方法效果差不多，INTER_AREA的图像对比度最低，INTER_LANCZOS4对比度最高
       INTER_CUBIC：
       INTER_LANCZOS4： 
    """

    img_origin = cv.imread("pictures/interpolation_origin.jpg")
    img_shrink = cv.resize(img_origin, (195,261), interpolation=cv.INTER_NEAREST)


    img_shrink_INTER_NEAREST  = cv.resize(img_origin, (195,261), interpolation=cv.INTER_NEAREST)
    img_enlarge_INTER_NEAREST  = cv.resize(img_shrink, (391,523), interpolation=cv.INTER_NEAREST)

    img_shrink_INTER_CUBIC  = cv.resize(img_origin, (195,261), interpolation=cv.INTER_CUBIC )
    img_enlarge_INTER_CUBIC  = cv.resize(img_shrink, (391,523), interpolation=cv.INTER_CUBIC )

    img_shrink_INTER_AREA = cv.resize(img_origin, (195,261), interpolation=cv.INTER_AREA)
    img_enlarge_INTER_AREA = cv.resize(img_shrink, (391,523), interpolation=cv.INTER_AREA)

    img_shrink_INTER_LANCZOS4  = cv.resize(img_origin, (195,261), interpolation=cv.INTER_LANCZOS4 )
    img_enlarge_INTER_LANCZOS4 = cv.resize(img_shrink, (391,523), interpolation=cv.INTER_LANCZOS4 )

 
    cv.imshow('img_origin', img_origin)
    cv.imshow('img_shrink_INTER_NEAREST', img_shrink_INTER_NEAREST)
    cv.imshow('img_enlarge_INTER_NEAREST', img_enlarge_INTER_NEAREST)

    cv.imshow('img_shrink_INTER_CUBIC', img_shrink_INTER_CUBIC)
    cv.imshow('img_enlarge_INTER_CUBIC', img_enlarge_INTER_CUBIC)

    cv.imshow('img_shrink_INTER_AREA', img_shrink_INTER_AREA)
    cv.imshow('img_enlarge_INTER_AREA', img_enlarge_INTER_AREA)

    cv.imshow('img_shrink_INTER_LANCZOS4', img_shrink_INTER_LANCZOS4)
    cv.imshow('img_enlarge_INTER_LANCZOS4', img_enlarge_INTER_LANCZOS4)

    cv.waitKey(0)


if __name__=="__main__":  
    interpolation()