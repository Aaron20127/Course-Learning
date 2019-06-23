"""
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf
问题：  1.在LM算法中将下降方向d加了一个负号后函数才收敛，而原始求出的方向却会使
           函数值增大，不知是算法问题还是求导方程出了问题。
           d = -1 * np.dot( la.inv(np.dot(A.T, A) + alpha*np.eye(A.shape[1])), np.dot(A.T, f))
        2.单应矩阵的右下角为啥一直为0
"""


import cv2 as cv
import os
import sys
import numpy as np
import copy
from numpy import linalg as la
from matplotlib import pyplot as plt
import Levenberg_Marquardt as LMS
import scipy.io as scio

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../../common")
# change the work path
os.chdir(abspath)

import misc_utils
import math_utils
import glob


def calculateHomography(list_world, list_image):
    """
    描述: 张正友标定法，使用LM算法最小化非线性二次方程。总共有9个参数。
          函数的收敛性和初始值关系特别大。这里的单应变换是二维平面到二维平面的变换，
          因为假设的标定板平面的为xoy平面。
    参数: c_world: 世界坐标，list的每个元素代表一幅图片对应的世界坐标，list[narray]
          c_image: 图片坐标，list的每个元素代表一幅图片，list[narray]
    返回: 参数的初始估计值
    结果：得到了和cv.findHomography()一样的参数
    问题: 
          1.在LM算法中将下降方向d加了一个负号后函数才收敛，而原始求出的方向却会使
           函数值增大，不知是算法问题还是求导方程出了问题。
           d = -1 * np.dot( la.inv(np.dot(A.T, A) + alpha*np.eye(A.shape[1])), np.dot(A.T, f))
    """
    list_H=[]
    for i in range(len(list_world)):
        c_world = list_world[i]
        c_image = list_image[i]

        # 1.求齐次矩阵Ax=0，即H的初始值
        A = np.zeros((2*c_world.shape[0], 9),dtype=np.float64) 
        for i in range(c_world.shape[0]):
            j=2*i
            A[j][0]=c_world[i][0]
            A[j][1]=c_world[i][1]
            A[j][2]=1
            A[j][6]=-c_world[i][0]*c_image[i][0]
            A[j][7]=-c_world[i][1]*c_image[i][0]
            A[j][8]=-c_image[i][0]

            A[j+1][3]=c_world[i][0]
            A[j+1][4]=c_world[i][1]
            A[j+1][5]=1
            A[j+1][6]=-c_world[i][0]*c_image[i][1]
            A[j+1][7]=-c_world[i][1]*c_image[i][1]
            A[j+1][8]=-c_image[i][1]

        # SVD分解得到最小奇异值对应的特征向量作为初始值，最接近齐次方程的解
        # dataNew = 'H.mat'
        # scio.savemat(dataNew, {'A':A})
        # B = scio.loadmat(dataNew)

        U,sigma,VT=la.svd(A)
        print(VT[-1].reshape((3,3)) / VT[-1][8])

        H = LMS.nonlinearLeastSquare_LM(\
            VT[-1], \
            LMS.LM_findHomography(c_world, c_image), \
            alpha=0.01, beta=1000.0, e=0.000001, op=True)

        list_H.append(H)

    return list_H

def calculateCameraParameters(c_world, c_image):
    """
    描述: 张正友标定法，使用LM算法最小化非线性二次方程。总共有21个参数。
          初始解使用分解单应矩阵的方法获得。函数的收敛性和初始值关系特别大。
    参数: c_world: 世界坐标，list的每个元素代表一幅图片对应的世界坐标，list[narray]
          c_image: 图片坐标，list的每个元素代表一幅图片，list[narray]
    返回: 参数的初始估计值
    结果：收敛到一个局部最优
    问题：1.初始值貌似还不是很准确
          2.在LM算法中将下降方向d加了一个负号后函数才收敛，而原始求出的方向却会使
           函数值增大，不知是算法问题还是求导方程出了问题。
           d = -1 * np.dot( la.inv(np.dot(A.T, A) + alpha*np.eye(A.shape[1])), np.dot(A.T, f))
    """
    def v(i, j, M):
        m = np.array([\
                      M[i-1][1-1]*M[j-1][1-1],\
                      M[i-1][1-1]*M[j-1][2-1] + M[i-1][2-1]*M[j-1][1-1],\
                      M[i-1][2-1]*M[j-1][2-1],\
                      M[i-1][3-1]*M[j-1][1-1] + M[i-1][1-1]*M[j-1][3-1],\
                      M[i-1][3-1]*M[j-1][2-1] + M[i-1][2-1]*M[j-1][3-1],\
                      M[i-1][3-1]*M[j-1][3-1]]) 
        return m

    M, mask = cv.findHomography(c_world[:,:2], c_image)

    # 根据 h1.T*A.-T*A-1*h2 = 0 和 h1.T*A.-T*A-1*h2 = 0 求B = A.-T*A-1
    V = np.zeros((2,6))
    V[0] = v(1,2,M)
    V[1] = v(1,1,M) - v(2,2,M)

    U,sigma,VT=la.svd(V)
    B11, B12, B22, B13, B23, B33 = VT[-1]

    # 计算内参
    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lgda = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = (lgda / B11)**(0.5)
    beta = -1*abs((lgda*B11/(B11*B22 - B12**2)))**(0.5)
    gamma = -B12*alpha**2*beta / lgda
    u0 = gamma*v0/beta - B13*alpha**2/lgda

    # 计算外参
    A = np.array([[B11, B12, B13],\
                  [B12, B22, B23],\
                  [B13, B23, B33]])
    A_I = la.inv(A)
    h1 = M[:,0].reshape((3,1))
    h2 = M[:,1].reshape((3,1))
    h3 = M[:,2].reshape((3,1))

    longda = 1 / la.norm(np.dot(A_I, h1))
    r1 = longda*np.dot(A_I, h1).flatten()
    r2 = longda*np.dot(A_I, h2).flatten()
    r3 = np.cross(r1, r2)
    t = longda*np.dot(A_I, h3).flatten()

    # 对参数赋初始值
    R = np.zeros((3,3))
    R[:,0] = r1
    R[:,1] = r2
    R[:,2] = r3

    ret = np.ones((21,1)).flatten()

    ret[0:9] = R.flatten()
    ret[9:12] = t
    ret[17] = alpha
    ret[18] = beta
    ret[19] = u0
    ret[20] = v0

    # 下边这个是cv函数求出的参数
    # ret[0:9] = [ -4.52375672e-04, -9.98131298e-01,  6.11040672e-02,
    #               9.25516939e-01,  2.27225791e-02,  3.78023916e-01,
    #              -3.78705944e-01,  5.67238581e-02,  9.23777144e-01]
    # ret[9:12] = [2.30732174,-5.2224209,13.24357609]
    # ret[12:17] = [-0.23515635,-0.48019479,1.05075903,-0.02963008,0.00905709]

    # ret[17] = 453.57367348
    # ret[18] = 473.04923693
    # ret[19] = 162.57328512
    # ret[20] = 221.5293728

    # 使用LM算法求解相机参数
    EI = LMS.nonlinearLeastSquare_LM(\
        ret, \
        LMS.LM_findIntrinsicAndExtrinsicParameters(c_world, c_image), \
        alpha=0.01, beta=10.0, e=0.0001, op=True)

    return ret

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
        # Find the chess board corners 必须两黑两白交错才是角点，如果没有检测到指定个数的角点，则返回0
        ret, corners = cv.findChessboardCorners(gray, (corner_row,corner_col), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # cv.drawChessboardCorners(img, (corner_row,corner_col), corners2, ret)
            # cv.imshow(fname, img)
            # cv.waitKey(0)

            objpoints.append(objp)
            imgpoints.append(corners2.reshape(corner_row * corner_col,2))
    
    # 使用LM算法计算单应矩阵
    H = calculateHomography(objpoints, imgpoints)
    # x = calculateCameraParameters(objpoints[0], imgpoints[0])

    cv.waitKey(0)
    cv.destroyAllWindows()

calibrateCamera()
