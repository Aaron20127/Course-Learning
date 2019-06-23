"""
这个文件主要是按照官网的教程对相机进行标定和矫正
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

问题：   1.初始值貌似还不是很准确
        2.在LM算法中将下降方向d加了一个负号后函数才收敛，而原始求出的方向却会使
        函数值增大，不知是算法问题还是求导方程出了问题。
        d = -1 * np.dot( la.inv(np.dot(A.T, A) + alpha*np.eye(A.shape[1])), np.dot(A.T, f))
"""

import os
import cv2 as cv
import sys
import numpy as np
import copy
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../../common")
# change the work path
os.chdir(abspath)

import misc_utils
import glob


def calibrateCamera():
    corner_row=9
    corner_col=6
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((corner_row*corner_col,3), np.float32)
    objp[:,:2] = np.mgrid[0:corner_row,0:corner_col].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.jpg')

    for fname in images:
        img = cv.imread(fname)
        cv.imshow(fname, img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners 必须两黑两白交错才是角点，如果没有检测到指定个数的角点，则返回0
        ret, corners = cv.findChessboardCorners(gray, (corner_row,corner_col), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners，并用斜线绘制出最后一行到第一行的移动方向,红色的点是起始点
            cv.drawChessboardCorners(img, (corner_row,corner_col), corners2, ret)
            # cv.imwrite('drawChessboardCorners.jpg',img)
            cv.imshow('img', img) # 必须加一个imshow()才能把角点画出来

            ret, mtx, dist, rvecs, tvecs = \
                cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            R, Jacbi= cv.Rodrigues(rvecs[0])
            print("\nret: \n", ret)
            print("\nmtx: \n", mtx)
            print("\ndist: \n", dist)
            print("\nrvecs: \n", rvecs)
            print("\ntvecs: \n", tvecs)
            print("\nR:\n", R)
            print("\nJacbi:\n", Jacbi)
            
            ## 使用getOptimalNewCameraMatrix
            img2rectify = cv.imread(fname)
            h, w = img2rectify.shape[:2]
            # alpha的取值范围为0-1，当为0时，undistort后图像的黑边将完全消失；
            # 当为1时，undistort后图像的黑边将完全保留；
            # 黑边是由于矫正后的像素移动到其他位置造成的。
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # 1.使用undistort
            dst = cv.undistort(img2rectify, mtx, dist, None, newcameramtx)
            cv.imshow('undistort uncrop newcameramtx ' + fname, dst)

            # crop the image, 将多余的黑色区域截去
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv.imshow('undistort crop newcameramtx ' + fname, dst)

            # 未使用getOptimalNewCameraMatrix
            dst = cv.undistort(img2rectify, mtx, dist, None)
            cv.imshow('undistort no newcameramtx ' + fname, dst)

            # 2.remap
            img2rectify = cv.imread(fname)
            mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv.remap(img2rectify, mapx, mapy, cv.INTER_LINEAR)
            cv.imshow('remap uncrop ' + fname, dst)

            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv.imshow('remap crop ' + fname, dst)
            # misc_utils.plot().plot_picture([img[...,::-1]], title=['hot'])
            # plt.show()
    
            # 3.重投影误差，所有误差的二范数的平方的和开方，再计算所有点的平均值
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
                mean_error += error
            print( "mean error: {}".format(mean_error/len(objpoints)) )

        cv.waitKey(0)
        cv.destroyAllWindows()


calibrateCamera()


