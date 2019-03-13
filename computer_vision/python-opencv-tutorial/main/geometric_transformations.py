"""
对2D平面进行透视变换。
透视变换包括：
    1.（相似变换）平移，旋转，放缩
    2.（仿射变换）平行边保持平行
    3.（透视变换）直线边始终是直线
所以 透视 > 仿射 > 相似
"""

import cv2 as cv
import sys
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

sys.path.append("D:/common") 
import misc_utils

def plot(img, points, text, text_pos, fontScale=0.5):
    """
    描述: 绘制多边形和字符
    参数: img: 图片矩阵，narray, shape(m,n,3)
          points: 要画到图形中的点，是一个列表，列表中是narray，shape(m,1,2)
          text: 字符串名称，string
          text_pos: 字符串的位置
          fontScale: 字符串大小，float
    返回: None
    """
    cv.polylines(img, points, True, (0,255,255), thickness=2)
    putText(img, text, text_pos, fontScale)

def putText(img, text, pos, fontScale=0.5):
    """
    描述: 字符
    参数: img: 图片矩阵，narray, shape(m,n,3)
          text: 字符串名称，string
          text_pos: 字符串的位置
          fontScale: 字符串大小，float
    返回: None
    """
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX,\
               fontScale, (255,255,255), 1, cv.LINE_AA)

def similarityTransformation_2D(points, s, theta, t):
    """
    描述: 2D坐标相似变换
    参数: points: 图形的坐标点，每行一个坐标，narray, shape(m,2)
          s: 图像的放大倍数，float
          theta: 旋转角度，弧度表示
          t: 平移，x方向和y方向，list
    返回: 相似变换后的新坐标，narray, shape(m,1,2)，dtype="int32"
    """
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    ts = np.array(t).reshape((2,1))

    ret =  s * np.dot(R, points.T) + ts
    return ret.T.reshape((-1,1,2)).astype("int32")

def affineTransformation_2D(points, A, t):
    """
    描述: 2D坐标仿射变换
    参数: points: 图形的坐标点，每行一个坐标，narray, shape(m,2)
          A: 仿射变换矩阵，以列表示，list，len 4
          t: 平移，x方向和y方向，list，len 2
    返回: 仿射变换后的新坐标，narray, shape(m,1,2)，dtype="int32"
    """
    affine = np.array(A).reshape((2,2))
    ts = np.array(t).reshape((2,1))

    ret =  np.dot(affine, points.T) + ts
    return ret.T.reshape((-1,1,2)).astype("int32")

def perspectiveTransformation_2D(points, A, t):
    """
    描述: 2D坐标仿射变换
    参数: points: 图形的坐标点，每行一个坐标，narray, shape(m,2)
          A: 仿射变换矩阵，以列表示，list，len 6
          t: 平移，x方向和y方向，list，len 2
    返回: 透视变换后的新坐标，narray, shape(m,1,2)，dtype="int32"
    """
    affine = np.array(A).reshape((3,2))
    ts = np.array([[t[0]],[t[1]],[1]])

    H = np.hstack((affine, ts))
    x = np.vstack((points.T, np.ones((1, points.shape[0]))))
    X =  np.dot(H, x)

    ret = np.zeros((2, X.shape[1]))
    for c in range(X.shape[1]):
        ret[0,c] = X[0,c] / X[2,c]
        ret[1,c] = X[1,c] / X[2,c]

    return ret.T.reshape((-1,1,2)).astype("int32")


def testTransformation_2D():
    """
    描述: 简单展示相似，仿射，和透视变换
    返回: None
    """
    # 初始矩形坐标和画布
    origin = np.array([[0,0],[20,0], [40,0],[40,20], [40,40], [20,40], [0,40]], np.int32)
    img = np.zeros((512,512,3), np.uint8) 

    # 相似变换 平移+旋转+缩放
    p0  = similarityTransformation_2D(origin,  1,   0, (40, 100))   # 平移
    p10 = similarityTransformation_2D(origin,  1.5, 0, (130, 100))  # 平移+放大1.5
    p11 = similarityTransformation_2D(origin,  1,   np.pi/4, (160, 190)) # 平移+旋转45
    p12 = similarityTransformation_2D(origin,  1.2, np.pi/4, (160, 270)) # 平移+放大1.2+旋转

    # 仿射变换
    p20 = affineTransformation_2D(origin, (1,0.3,1,1), (250, 100)) # 平移+仿射
    p21 = affineTransformation_2D(origin, (1,1,1,-0.4), (240, 220)) # 平移+仿射

    # 透视变换
    p30 = perspectiveTransformation_2D(origin, (3,1,1,4,0.005,0.005), (380, 100))

    plot(img, [p0], 'origin', (40, 60))
    plot(img, [p10, p11, p12], 'similarity', (125, 60))
    plot(img, [p20, p21], 'affine', (250, 60))
    plot(img, [p30], 'perspective', (350, 60))

    cv.imshow("img", img)
    cv.waitKey(0)


def adjustTransformationParameters(graphics_points):
    """
    描述: 使用trackbar可以对透视变换矩阵的参数实时调整，以实现各种变换的显示
    参数: graphics_points: 图形的坐标点，每行一个坐标，narray, shape(m,2)
    返回: None
    """
    def nothing(x):
        """控件回调函数
        """
        pass

    imgTrackbar = np.zeros((200,400,3), np.uint8) #大的黑板，保证拖动条能完全显示
    img = np.zeros((720,600,3), np.uint8) 

    cv.namedWindow('image', 0)
    cv.createTrackbar('a11','image',0,1000, nothing) #1000个参数
    cv.createTrackbar('a12','image',500,1000, nothing) #1000个参数
    cv.createTrackbar('a21','image',500,1000, nothing) #1000个参数
    cv.createTrackbar('a22','image',0,1000, nothing) #1000个参数
    cv.createTrackbar('a31','image',500,1000, nothing) #1000个参数
    cv.createTrackbar('a32','image',500,1000, nothing) #1000个参数
    cv.createTrackbar('t1','image',414,800, nothing) #1000个参数
    cv.createTrackbar('t2','image',447,800, nothing) #1000个参数
    cv.imshow('image',imgTrackbar)

    
    while(1):
        cv.imshow('image',imgTrackbar)
        cv.imshow('img',img)
        cv.waitKey(1)

        img = np.zeros((720,600,3), np.uint8) #大的黑板，保证拖动条能完全显示

        a11 = (cv.getTrackbarPos('a11','image') - 500) / 100
        a12 = (cv.getTrackbarPos('a12','image') - 500) / 100
        a21 = (cv.getTrackbarPos('a21','image') - 500) / 100
        a22 = (cv.getTrackbarPos('a22','image') - 500) / 100
        a31 = (cv.getTrackbarPos('a31','image') - 500) / 10000
        a32 = (cv.getTrackbarPos('a32','image') - 500) / 10000        
        t1 = cv.getTrackbarPos('t1','image')
        t2 = cv.getTrackbarPos('t2','image')

        
        # p = perspectiveTransformation_2D(origin, (3,1,1,4,0.005,0.005), (380, 100))
        p = perspectiveTransformation_2D(graphics_points, (a11,a12,a21,a22,a31,a32), (t1, t2))
        plot(img, [p], 'Perspective Transformation:', (10, 20), 0.7)
        putText(img, '[%.4f  %.4f  %.4f' % (a11,a12,t1), (10, 60))
        putText(img, ' %.4f  %.4f  %.4f' % (a21,a22,t2), (10, 80))
        putText(img, ' %.4f  %.4f  %.4f]' % (a31,a32,1), (10, 100))

    cv.destroyAllWindows()

def generatingCircleCoordinates(r, num, center=(0,0)):
    """
    描述: 生成圆形坐标
    参数: r: 圆形半径, float
          num: 产生多少个点
          center：圆形中心，list 2
    返回: 圆形坐标，narray, shape(m,2)
    """
    total_deta = math.pi
    deta = np.linspace(-total_deta, total_deta, num, endpoint=False)
    cx = r * np.cos(deta) + center[0] 
    cy = r * np.sin(deta) + center[1]
    ret = np.hstack((cx.reshape((num,1)), cy.reshape((num,1))))

    return ret

if __name__=="__main__":
    ## 1.展示各种变换
    # testTransformation_2D()

    ## 2.实时调整透视变换矩阵，观察图像变化
    #  步骤：1）先将a12,a21,a31,a32尽量调整到0，这时与原图像才比较接近
    #        2）然后调整t1,t2将图像移动到画布中间
    #        3）任意调整参数，观察变化，可以以相似->仿射->透视的顺序调整
    square = np.array([[0,0],[20,0], [40,0],[40,20], [40,40], [20,40], [0,40]], np.int32)
    # circle = generatingCircleCoordinates(20, 200)
    adjustTransformationParameters(square)
