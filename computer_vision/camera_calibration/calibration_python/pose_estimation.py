"""
pose estimation
"""
import sys
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
import glob


# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../../common")
# change the work path
os.chdir(abspath)


def drawAixs(img, corners, imgpts):
    ## The first corner is connected to three axes
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def pose_estimation():
    corner_row=7 
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
        ### 1. calibration
        img = cv.imread(fname)
        cv.imshow(fname, img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners 必须两黑两白交错才是角点，如果没有检测到指定个数的角点，则返回0
        ret, corners = cv.findChessboardCorners(gray, (corner_row,corner_col), None)
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        cv.drawChessboardCorners(img, (corner_row,corner_col), corners2, ret)
        cv.imshow('img', img) 


        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            ret, mtx, dist, rvecs1, tvecs1 = \
                cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            ### 2.get the external parameters and project points
            axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)
            ## Find the rotation and translation vectors. get extern para
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            ## project 3D points to image plane, get the external parameters
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = drawAixs(img, corners2, imgpts)
            cv.imshow('img',img)
            cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == "__main__":
    pose_estimation()