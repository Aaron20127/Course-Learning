import time
import cv2 as cv
import os
import sys
import numpy as np
import copy
from numpy import linalg as la
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../common")
# change the work path
os.chdir(abspath)

import opencv
import misc_utils

def main():
    ### 1.矫正图像
    # read img
    filename =  abspath + '/box2.png'
    img = cv.imread( filename )

    # intrinsic parameters and distort
    mtx = np.array([[2.6182e+03,          0, 973.1464],
                    [         0, 2.6228e+03, 571.7691],
                    [         0,          0,        1]])
    dist = np.array([[-0.5222, -0.2738, 0, 0, 0]])
    
    # undistort
    undistort = opencv.cameraUnistort(mtx, dist, (img.shape[1], img.shape[0]))
    dst = undistort.undistort(img)

    cv.imwrite(abspath + "/box2_undistort.png", dst)

    ### 2.求出3D检测的边缘框，只要xoy平面框

    ### 3.将网格透视变换到整个图片，方便计算
    # 下边的所有坐标都是(col,row)，因为列方向是x坐标
    # row, col = dst.shape[:2]
    # pts1 = np.float32([[1020,457], [1007,967], [1790,977], [1750,453]])
    # pts2 = np.float32([[0,0], [0, row-1], [col-1, row-1], [col-1, 0]])
    # M = cv.getPerspectiveTransform(pts1,pts2)  # 必须是4个点对，求8个参数
    # image_pt = cv.warpPerspective(dst, M, (col,row))

    # data = [[image_pt[...,::-1], 'image_pt']]
    # misc_utils.plot().plot_multiple_picture((1,1), data, 'demo')
    # plt.show()

    ### 4.将3D检测的框四个顶点取平均求得车的中心位置

    # cv.namedWindow('img',0)
    # cv.namedWindow('dst',0)
    # cv.imshow('img', img)
    # cv.imshow('dst', dst)
    # cv.waitKey(0)


if __name__ == "__main__":
    main()