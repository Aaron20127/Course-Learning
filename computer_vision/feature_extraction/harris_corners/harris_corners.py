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

def harris_corners(img, blockSzie=2, SobelSize=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        blockSzie: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((blockSzie, blockSzie))

    response = np.zeros((H, W))

    dx = cv.Sobel(img,cv.CV_64F,1,0,ksize=SobelSize)
    dy = cv.Sobel(img,cv.CV_64F,0,1,ksize=SobelSize)

    ### YOUR CODE HERE
    dx_squared = dx ** 2
    dy_squared = dy ** 2
    dx_dy_multiplied = dx * dy

    half_blockSzie = blockSzie//2
    M = np.zeros((2,2))

    # response要跟图像一样大小，只计算出中心那些可以加窗的，边缘那些设置为0
    k=0
    for i in range(H-blockSzie+1):
        for j in range(W-blockSzie+1):
            M[0,0] = np.sum(window * dx_squared[i:i + blockSzie, j:j + blockSzie])
            M[0,1] = np.sum(window * dx_dy_multiplied[i:i + blockSzie, j:j + blockSzie])
            M[1,0] = M[0,1]
            M[1,1] = np.sum(window * dy_squared[i:i + blockSzie, j:j + blockSzie])

            R = np.linalg.det(M) - k * (np.trace(M)**2)

            i_shifted = i + half_blockSzie
            j_shifted = j + half_blockSzie
            response[i_shifted][j_shifted] = R

            k +=1
            print((H-blockSzie+1) * (W-blockSzie+1), '/', k)
    ### END YOUR CODE

    return response


    
if __name__ == "__main__":
    
    filename = data_path + '/chessboard.png'
    img = cv.imread( filename )
    img = cv.resize(img, (int(img.shape[0]*0.02), int(img.shape[1]*0.02)))
    img = img[0:35, 0:35]

    gray = cv.cvtColor( img, cv.COLOR_BGR2GRAY )
    gray = np.float32( gray )

    # cv.imshow("img", img)
    # cv.imshow("gray", gray)
    # cv.waitKey(0)

    # harris detect
    img_corner = harris_corners(gray, 2)      

    # imshow
    data = [
            [gray, 'origin', 'gray'],
            [img_corner, 'harris corners', 'gray']
    ]
    misc_utils.plot().plot_multiple_picture((1,2), data)
    plt.show()