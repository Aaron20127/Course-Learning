"""
这个文件是处理标定后的矫正问题，矫正图片时，只使用相机内参和畸变系数就可以实现。
存在问题：1.初始值貌似还不是很准确
         2.在LM算法中将下降方向d加了一个负号后函数才收敛，而原始求出的方向却会使
           函数值增大，不知是算法问题还是求导方程出了问题。
           d = -1 * np.dot( la.inv(np.dot(A.T, A) + alpha*np.eye(A.shape[1])), np.dot(A.T, f))
"""

import cv2 as cv
import sys
import numpy as np
import copy
from numpy import linalg as la
from matplotlib import pyplot as plt
import Levenberg_Marquardt as LMS

sys.path.append("D:/common") 
import misc_utils
import math_utils
import glob

def undistort(img, mtx, dist):
    """
    描述: 使用标定后的内参和畸变系数对图片的每个像素的坐标重新映射，得到映射矩阵
    参数: img: 原始图片，narray
          mtx: 相机内参矩阵, 最简单形式，只有fx,fy,u0,v0,narray, shape(3,3)
          dist: 畸变系数k1,k2,p1,p2,k3，narray, shape(,5)
    返回: 重映射矩阵，矩阵的每个元素值是该元素对应的新的位置，narray, shape(m,n)
    问题：1.初始值貌似还不是很准确
         2.在LM算法中将下降方向d加了一个负号后函数才收敛，而原始求出的方向却会使
           函数值增大，不知是算法问题还是求导方程出了问题。
           d = -1 * np.dot( la.inv(np.dot(A.T, A) + alpha*np.eye(A.shape[1])), np.dot(A.T, f))
    """
    import time
    remap_mtx = np.zeros((img.shape[0], img.shape[1], 2))
    fx = mtx[0][0]
    fy = mtx[1][1]
    u0 = mtx[0][2]
    v0 = mtx[1][2]
    time_start = 0
    time_end  = 0
    for u in range(img.shape[1]):
        # print("\r", "remap (u,v) ==> {:.2f}% {:.10f}s".\
        #     format(u/img.shape[1], time_end - time_start), end="")
        print("remap (u,v) ==> {:.2f}% {:.10f}s".\
            format(u/img.shape[1], time_end - time_start))
        time_start = time.time()
        for v in range(img.shape[0]):
            # 给未畸变的归一化图片坐标赋初值，因为畸变位置到未畸变的位置不远，
            # 所以使用畸变位置u,v作为未畸变位置的初始值，计算归一化xx,yy的初始值
            xx = (u - u0) / fx 
            yy = (v - v0) / fy

            norm_xy = LMS.nonlinearLeastSquare_LM(\
                np.array([xx,yy]), \
                LMS.LM_undistort(mtx, dist, np.array([u, v])), \
                alpha=0.01, beta=10.0, e=0.1, op=False)
            
            remap_mtx[v][u][0] = norm_xy[0]*fx + u0
            remap_mtx[v][u][1] = norm_xy[1]*fy + v0
        time_end = time.time()

    np.save("undistort.npy", remap_mtx)
    return remap_mtx

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
        # cv.imshow(fname, img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners 左上角为原点，从左往右为x轴，从上到下为y轴
        ret, corners = cv.findChessboardCorners(gray, (corner_row,corner_col), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            # 子采样精确角点坐标
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # cv.drawChessboardCorners(img, (corner_row,corner_col), corners2, ret)
            # cv.imshow(fname, img)
            # cv.waitKey(0)
            # 物体世界坐标
            objpoints.append(objp)
            # 图片坐标
            imgpoints.append(corners2.reshape(corner_row * corner_col,2))
            # 相机标定得到外参和内参
            ret, mtx, dist, rvecs, tvecs = \
                cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # 根据内参和畸变系数矫正图片，使用LM算法将图片坐标映射到新的坐标，并将映射矩阵保存
            remap_mtx = undistort(img, mtx, dist[0])
            # remap_mtx = np.load('undistort.npy')

            # 获取最大最小像素位置
            maxn = int(np.max(remap_mtx))+1    
            minn = -1*int(np.min(remap_mtx))+1

            # 创建一个新的图
            rectified_img = np.zeros((maxn + minn, maxn + minn), dtype='uint8') 
            # 将畸变的图像的像素值移动到未发生畸变时的图像位置
            for u in range(img.shape[1]):
                for v in range(img.shape[0]):
                    x, y = np.rint(remap_mtx[v][u] + minn) # 将像素位置为负的值变成正值
                    rectified_img[int(y)][int(x)] = gray[v][u]

            # 将没有像素的黑边去掉
            min_row = -1
            max_row = 0
            min_col = -1
            max_col = 0
            for i in range(rectified_img.shape[0]):
                for j in range(rectified_img.shape[1]):
                    if rectified_img[i][j]:
                        if min_row == -1:
                            min_row = i
                        if min_col == -1:
                            min_col = j

                        if i < min_row:
                            min_row = i
                        if j < min_col:
                            min_col = j
                        if i > max_row:
                            max_row = i
                        if j > max_col:
                            max_col = j

            new_rectified_img = rectified_img[min_row:(max_row+1),min_col:(max_col+1)]
                        
   
            ## 自己的矫正方法。将图片矫正后，有的地方没有像素，形成黑色条纹，需要使用适当的方法将其填充
            ## 黑色条纹表征畸变的分布
            cv.imshow("img", img)
            cv.imshow("rectified_img", rectified_img)
            cv.imshow("new_rectified_img", \
                cv.resize(new_rectified_img,(int(new_rectified_img.shape[1]/1.1),\
                          int(new_rectified_img.shape[0]/1.1)),interpolation=cv.INTER_AREA))


            ## opencv的矫正方法
            h, w = img.shape[:2]
            # 使用下边这个函数可以将原像素的所有信息保存下来
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            cv.imshow('opencv distort', dst)

    cv.waitKey(0)
    cv.destroyAllWindows()


calibrateCamera()
