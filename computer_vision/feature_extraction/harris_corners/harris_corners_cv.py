import cv2 as cv
import os
import sys
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
common_path = abspath + "/../../../common"
data_path = common_path + "/data"
# add the library file path
sys.path.append(common_path)
# change the work path
os.chdir(abspath)

import misc_utils
import math_utils
import opencv

def cornerHarris():
    filename = [ "corner.png",
                 "edge.png",
                 "flat.png",
                 "bevel_edge.png",
                 "bevel_edge_1.png",
                 "blackboard.png"]

    # 控件
    button_units = [['windowSize',  2, 50],
                    ['gradientSize', 1, 15],
                    ['k', 4, 6]]
    track_bar = opencv.createTrackbar(button_units, "trackbar", block=True)

    # 读取文件
    filename1 =  'tmp/flower.jpg'
    filename2 =  'tmp/stinkbug.png'
    img_1 = cv2.imread( filename1 )
    img_2 = cv2.imread( filename2 )


    while(True):


    # filename = data_path + '/chessboard.png'
    # filename = "test.png"
    filename = abspath + "/bevel_edge.png"
    img = cv.imread( filename )
    # img = cv.resize(img, (int(img.shape[0]*0.02), int(img.shape[1]*0.02)))
    # img = img[0:35, 0:35]
    gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
    gray = np.float32( gray )

    # find Harris corners
    dst = cv.cornerHarris( gray, 2, 3, 0.04 )
    dst = dst / dst.max()  


    # draw corners
    img_corners = img.copy()
    img_corners[dst>0.01 * dst.max()] = [0,0,255]

    

    # imshow
    data = [ [img[...,::-1], 'Original'],
             [dst, 'harris corners', 'gray'],
             [img_corners[...,::-1], 'img add corners'],]

    misc_utils.plot().plot_multiple_picture((1,3), data)
    plt.show()


def cornerWithSubpixelAccuracy():
    filename = data_path + '/blox.jpg'
    img = cv.imread( filename )
    gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
    gray = np.float32( gray )

    # find Harris corners
    dst = cv.cornerHarris( gray, 2, 3, 0.04 )
    #result is dilated for marking the corners, not important
    dst = cv.dilate( dst, None )
    ret, dst = cv.threshold( dst, 0.01*dst.max(), 255, 0 )
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # define the criteria to stop refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)


    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:,1], res[:,0]] = [0,0,255]
    img[res[:,3], res[:,2]] = [0,255,0]

    # imshow
    data = [[img[...,::-1], 'Original']]

    misc_utils.plot().plot_multiple_picture((1,1), data)
    plt.show()


if __name__ == "__main__":
    cornerHarris()        
    