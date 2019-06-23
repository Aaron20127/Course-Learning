"""
对双目拍摄的矫正后的图片求视差
"""
import sys
import numpy as np
import cv2 
import os
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../../common")
# change the work path
os.chdir(abspath)


def createDisparity(img_L, img_R, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(img_L,img_R)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("numDisparities=%d, blockSize=%d" % (numDisparities, blockSize))
    plt.imshow(disparity,'gray')

def main():
    cones_img_1 = "cones/im2.png"
    cones_img_2 = "cones/im6.png"

    teddy_img_1 = "teddy/im2.png"
    teddy_img_2 = "teddy/im6.png"

    img_L = cv2.imread(cones_img_1,0)
    img_R = cv2.imread(cones_img_2,0)

    # img_L = cv2.imread(teddy_img_1,0)
    # img_R = cv2.imread(teddy_img_2,0)

    # for i in range(4, 6):
    #     disparitynum = i * 16
    #     for j in range(3, 6):
    #         block = 2*j - 1
    #         createDisparity(img_L, img_R, disparitynum, block)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img_L,'gray')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img_R,'gray')

    createDisparity(img_L, img_R, 64, 7)
    
    plt.show()

    # # 显示图片
    # cv2.imshow("img_R", img_R) 
    # cv2.imshow("img_rgb", img_rgb) 
    # cv2.imshow("img_binary_rgb", img_binary_rgb) 

    # cv2.imwrite("Bayesian_decision/img_gray.jpg", img_gray)
    # cv2.imwrite("Bayesian_decision/img_binary_gray.jpg", img_binary_gray)
    # cv2.imwrite("Bayesian_decision/img_rgb.jpg", img_rgb)
    # cv2.imwrite("Bayesian_decision/img_binary_rgb.jpg", img_binary_rgb)

    # cv2.waitKey (0)
    # cv2.destroyAllWindows()

if __name__=="__main__":
    main()