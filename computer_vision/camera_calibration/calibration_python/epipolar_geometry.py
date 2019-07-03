"""
get pipolar geometry
"""
import sys
import numpy as np
import cv2 as cv
import os
from numpy import linalg as la
from matplotlib import pyplot as plt
import glob


# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
common_path = abspath + "/../../../common"
data_path = common_path + "/data"
sys.path.append(common_path)
# change the work path
os.chdir(abspath)

import math_utils

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    row, col = img1.shape
    # in order to get 3 channels instead of color image
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)  
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    # r[0], [r1] and r[2] represent a, b and c respectively
    i = 1
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        if i > 0:
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ]) # first row
            x1,y1 = map(int, [col, -(r[2]+r[0]*col)/r[1] ]) # last row
            img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv.circle(img1,tuple(pt1),5,color,1)
            img2 = cv.circle(img2,tuple(pt2),5,color,1)
        i = i + 1

    return img1,img2
    

def main():
    ### 1. find best matchs
    img1 = cv.imread(data_path + '/left.jpg',0)  #queryimage # left image
    img2 = cv.imread(data_path + '/right.jpg',0) #trainimage # right image

    # img1 = cv.imread(abspath + '/images/left.jpg',0)  #queryimage # left image
    # img2 = cv.imread(abspath + '/images/right.jpg',0) #trainimage # right image

    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper, keep good matches
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt) 

    ### 2.Find the Fundamental Matrix.
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    ## Calculate F by myself, Effect is very poor, 8 point method？
    """M = np.ones((len(pts1), 9), dtype='int32')
    for i in range(len(pts1)):
        M[i][0:2] = pts1[i] * pts2[i][0] 
        M[i][2]   = pts2[i][0] 
        M[i][3:5] = pts1[i] * pts2[i][1] 
        M[i][5]   = pts2[i][1] 
        M[i][6:8] = pts1[i]
    
    # get F
    U, sigma, VT=la.svd(M)
    # for i in range(9):
    #     VT[i] = VT[i] / VT[i][8]
    F = VT[8].reshape(3,3)
    # print(F)

    # let rank = 2
    U, sigma, VT=la.svd(F)
    sigma[2] = 0
    F = np.dot(np.dot(U, np.diag(sigma)), VT)
    # print(F)"""

    
    # calculate fundenmental Matrix
    F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

    # We select only inlier points, the exeptional points are elimilated 
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    ### 3. Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image、
    # calculate left epilines: lr = (a b c).T = F.T * RightPoint
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

    plt.figure(),plt.imshow(img5)
    plt.figure(),plt.imshow(img3)
    plt.show()

if __name__ == "__main__":
    main()