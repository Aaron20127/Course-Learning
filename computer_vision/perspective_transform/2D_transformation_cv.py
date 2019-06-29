"""
对2D平面进行透视变换。
透视变换包括：
    1.（相似变换）平移，旋转，放缩
    2.（仿射变换）平行边保持平行
    3.（透视变换）直线边始终是直线
所以 透视 > 仿射 > 相似
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
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)

import misc_utils

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

def embedImage(img_canvas, img_new, pos=(0,0)):
    """
    描述: 将图片img_new嵌入到img_canvas的任意位置
    参数: img_canvas: 需要嵌入图像的图像矩阵, narray, shape(m,n,3)
          img_new: 需要嵌入img_canvas中的图像, narray, shape(m,n,3)
          pos：嵌入到img_canvas的位置，list 2，(x,y)
    返回: None 
    """
    img_canvas[pos[1]:img_new.shape[0]+pos[1], pos[0]:img_new.shape[1]+pos[0]] = img_new

def similarityTransformation_2D():
    """
    描述: 使用仿射函数做相似变换，比如拉伸和旋转
    函数：cv.getRotationMatrix2D，cv.warpAffine
    返回: None 
    """
    img_canvas = np.zeros((800,1600,3), np.uint8) 
    img = cv.imread('pictures/football.jpg')

    # 1.拉伸变换（相似变换）
    img = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
    embedImage(img_canvas, img, (100,100))
    putText(img_canvas, "resize", (180,300), fontScale=0.8)

    # 2.旋转变换（相似变换）
    rows = img.shape[0]
    cols = img.shape[1]
    # 以原图形中心旋转
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
    dst = cv.warpAffine(img,M,(cols,rows))
    embedImage(img_canvas, dst, (500,100))
    putText(img_canvas, "rotation", (600,300), fontScale=0.8)

    cv.imshow("img_canvas", img_canvas)
    cv.waitKey(0)

def affineTransformation_2D():
    """
    描述: 对图片做仿射变换，保平行线平行，仿射变换矩阵一共6个参数，需要3对映射坐标
    函数：cv.getAffineTransform, cv.warpAffine
    返回: None 
    """
    img = cv.imread('pictures/affine.jpg')
    rows,cols,ch = img.shape

    pts1 = np.float32([[65,65],[253,65],[65,253]])
    pts2 = np.float32([[20,150],[220,90],[200,310]])
    M = cv.getAffineTransform(pts1,pts2) # 必须是3个点对，求6个参数

    dst = cv.warpAffine(img,M,(cols,rows))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def perspectiveTransformation_2D():
    """
    描述: 对图片做透视变换时，保直线变换后还是直线，透视变换矩阵一共8个参数，需要4对映射坐标
          透视变换矩阵和相机求单应矩阵时，恰好四对映射点求出的变换矩阵时一样的
    函数：cv.getPerspectiveTransform，cv.warpPerspective
    返回: None 
    """
    img = cv.imread(abspath + '/pictures/perspective.jpg')
    rows,cols,ch = img.shape
    pts1 = np.float32([[66,77],[436,62],[33,456],[460,460]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    M = cv.getPerspectiveTransform(pts1,pts2) # 必须是4个点对，求8个参数
    # M, mask = cv.findHomography(pts1, pts2) # 透视投影矩阵应该就是单应矩阵，结果相同
    dst = cv.warpPerspective(img,M,(300,300))
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def adjustTransformationParameters():
    """
    描述: 实时调整4个映射坐标的位置，改变透视变换矩阵，以改变图像的相机视角。
    函数：cv.getPerspectiveTransform
    返回: None 
    """
    def nothing(x):
        """控件回调函数
        """
        pass

    imgTrackbar = np.zeros((100,600,3), np.uint8) #大的黑板，保证拖动条能完全显示
    img = cv.imread(abspath + '/pictures/perspective.jpg')
    imgBlock = np.zeros((720,600,3), np.uint8) 

    cv.namedWindow('image',0)
    cv.createTrackbar('p0x','image',0,400, nothing) #1000个参数
    cv.createTrackbar('p0y','image',0,400, nothing) #1000个参数
    cv.createTrackbar('p1x','image',400,400, nothing) #1000个参数
    cv.createTrackbar('p1y','image',0,400, nothing) #1000个参数
    cv.createTrackbar('p2x','image',0,400, nothing) #1000个参数
    cv.createTrackbar('p2y','image',400,400, nothing) #1000个参数
    cv.createTrackbar('p3x','image',400,400, nothing) #1000个参数
    cv.createTrackbar('p3y','image',400,400, nothing) #1000个参数
    cv.imshow('image',imgTrackbar)
    
    while(1):

        cv.imshow('image',imgTrackbar)
        cv.imshow('img',imgBlock)
        cv.waitKey(1)

        p0x = cv.getTrackbarPos('p0x','image')
        p0y = cv.getTrackbarPos('p0y','image')
        p1x = cv.getTrackbarPos('p1x','image')
        p1y = cv.getTrackbarPos('p1y','image')
        p2x = cv.getTrackbarPos('p2x','image')
        p2y = cv.getTrackbarPos('p2y','image')       
        p3x = cv.getTrackbarPos('p3x','image')
        p3y = cv.getTrackbarPos('p3y','image')

        ## 注意row是y, col是x，所以下边的所有坐标是(col, row)
        imgBlock = np.zeros((720,600,3), np.uint8) 
        pts1 = np.float32([[66,77],[436,62],[33,456],[460,460]])
        pts2 = np.float32([[p0x,p0y],[p1x,p1y],[p2x,p2y],[p3x,p3y]])
        M = cv.getPerspectiveTransform(pts1,pts2) # 必须是4个点对，求8个参数
        dst = cv.warpPerspective(img,M,(400,400))

        rows,cols,ch = dst.shape
        pos_x = 150
        pos_y = 120
        imgBlock[pos_x:rows+pos_x, pos_y:cols+pos_y] = dst

        putText(imgBlock, '[%.4f  %.4f  %.4f' % (M[0,0],M[0,1],M[0,2]), (10, 60))
        putText(imgBlock, ' %.4f  %.4f  %.4f' % (M[1,0],M[1,1],M[1,2]), (10, 80))
        putText(imgBlock, ' %.4f  %.4f  %.4f]' % (M[2,0],M[2,1],M[2,2]), (10, 100))

    cv.destroyAllWindows()

if __name__=="__main__":
    # similarityTransformation_2D()
    # affineTransformation_2D()
    # perspectiveTransformation_2D()
    adjustTransformationParameters()


