"""
各种各样函数
"""
import cv2 as cv
import os
import sys
import numpy as np
import copy
from matplotlib import pyplot as plt

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)

import misc_utils


def cvFilter2D():
    """描述：cv.filter2D是通用卷积滤波函数，测试时使用方形卷积核，对彩色图像使用归一和非归一化滤波
       注意：1. cv.filter2D是correlation而非convolution，convolution需要将kernel绕中心旋转180度
               具体参考：https://www.mathworks.com/help/images/what-is-image-filtering-in-the-spatial-domain.html#f16-20755
               https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
            2. 核的最大为11*11时使用的直接方法计算，大于11*11则使用其他方法计算
            3. 在图像边缘的地方使用核时，由于某些地方不存在像素，要使用插值方法
            4. 具体使用>>> help(cv.filter2D)命令查看使用方法
       归一化：滤波后的像素值是周围像素的平均值，这样不会改变图像的亮度（不知道对不对），只会模糊图片。
              核中权重值越大，图像变得越明亮，反之，图像越暗（也不知道对不对）。
       
    """
    img_BGR = cv.imread('pictures/flower.jpg')
    img_RGB = img_BGR[...,::-1]

    img_RGB[0:40,0:40] = img_RGB[80:120,100:140] # 截取一块像素到赋值到其他位置

    k1 = np.ones((4,4),np.float32)/16 # 归一化卷积核图像模糊
    k2 = np.ones((4,4),np.float32)/24 # 非归一化卷积核，貌似图像变暗

    # 边界使用 BORDER_DEFAULT 是使用镜像插值
    img_1 = cv.filter2D(img_RGB,-1,k1)
    img_2 = cv.filter2D(img_RGB,-1,k2)

    # img_RGB[:,0:120,1] = 10 # 对任意位置像素赋值

    data = [[img_RGB, 'Original'],
            [img_1, '4x4, 1/16'],
            [img_2, '4x4, 1/24']]

    misc_utils.plot().plot_multiple_picture((1,3), data)
    plt.show()

def cvBlur():
    """描述：直接使用归一化线性卷积核模糊图片，与cv.filter2D的归一化效果相同，
            不过使用cv.blur模糊图片更简单，因为只需要定义核的长和宽。
    """
    img_BGR = cv.imread('pictures/flower.jpg')
    img_RGB = img_BGR[...,::-1]

    blur2x2 = cv.blur(img_RGB,(2,2))
    blur4x4 = cv.blur(img_RGB,(4,4))
    blur9x9 = cv.blur(img_RGB,(9,9))

    data = [[img_RGB, 'Original'],
            [blur2x2, '2x2'],
            [blur4x4, '4x4'],
            [blur9x9, '9x9']]

    misc_utils.plot().plot_multiple_picture((2,2), data)
    plt.show()

def blending():
    """描述：使两幅图片混叠 g(x)=(1−α)f0(x)+αf1(x)
       cv函数：cv.addWeighted
       return: None
    """
    def nothing(x):
        """控件回调函数
        """
        print(cv.getTrackbarPos('alpha','image')/1000.0)
        pass

    img1 = cv.imread('pictures/flower.jpg')
    img2 = cv.imread('pictures/apple.jpg')

    img1=img1[0:170, 40:200] # 抠出图片中的像素
    img2=img2[0:170, 60:220]

    imgBlock = np.zeros((720,1080,3), np.uint8) #大的黑板，保证拖动条能完全显示

    cv.namedWindow('image')
    cv.createTrackbar('alpha','image',0,1000, nothing) #1000个参数
    
    while(1):
        cv.imshow('image',imgBlock)
        cv.waitKey(1)

        r1 = cv.getTrackbarPos('alpha','image')

        a1 = r1*0.001 # 两幅图片的比例权重和为1
        a2 = 1.0 - a1 
        img=cv.addWeighted(img1,a1,img2,a2,0)
        imgBlock[0:170, 40:200] = img #将图片嵌入黑板中

    cv.destroyAllWindows()

def addALogo():
    """描述：使用屏蔽罩将logo拼接到另一幅图上
       cv函数：cv.cvtColor, cv.threshold, cv.bitwise_not, cv.bitwise_and
       return: None
    """
    img1 = cv.imread('pictures/flower.jpg')
    img2 = cv.imread('pictures/opencv-logo-small.png')

    # 读取logo的分辨率，从大图中得到logo相同的区域
    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    # 将logo变成灰度图，然后设置阈值变成二值图（0-255）的mask，然后再将mask按位翻转得到mask_inv
    # 0表示黑色，255表示白色，mask的logo处是白色，mask_inv的logo处是黑色
    img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask) # 是将值按位取反

    cv.imshow('img2gray',img2gray)
    cv.imshow('mask',mask)      
    cv.imshow('mask_inv',mask_inv)
    cv.waitKey(0)

    # 将截取的彩图中logo的位置由于mask_inv是0，即直接生成黑色；
    # 非0的部分由于对应像素相同，按位与之后像素值不变
    img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv) # mask是二维矩阵
    cv.imshow('img1_bg',img1_bg)
    cv.waitKey(0)

    # 将原彩色图中的背景部分屏蔽成0，logo部分保留
    img2_fg = cv.bitwise_and(img2,img2,mask = mask)
    cv.imshow('img2_fg',img2_fg)
    cv.waitKey(0)

    # 将裁减出的logo和原图扣除logo的部分相加之后嵌入原图
    dst = cv.add(img1_bg,img2_fg)
    cv.imshow('dst',dst)
    img1[0:rows, 0:cols ] = dst
    cv.imshow('res',img1)
    cv.waitKey(0)
    cv.destroyAllWindows()

def HSV_valueRange():
    """描述：测试HSV的取值范围，当前系统的取值范围是H[0-179],S[0-254],V[1-255]
       位置：https://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html
       cv函数：cv.cvtColor
       return: None
    """
    list_HSV = [[],[],[]]
    imgBlock = np.zeros((1,1,3), np.uint8)
    for B in range(1,255+1):
        print("\r", "{:.2f}%".format(B/255.0), end="")
        for G in range(1,255+1):
            for R in range(1,255+1):
                imgBlock[:,:] = B,G,R
                hsv = cv.cvtColor(imgBlock, cv.COLOR_BGR2HSV)
                
                for i in range(3):
                    if hsv[0,0,i] not in list_HSV[i]:
                        list_HSV[i].insert(0,np.int(hsv[0,0,i]))
    
    for i in range(3):
        list_HSV[i].sort()
        print("",list_HSV[i])

    misc_utils.write_list_to_file('HSV_value_range.txt', list_HSV)

def trackingUsingHSV():
    """描述：在HSV色域中，过滤出图片中在某范围色彩值的像素，色相（红绿蓝），饱和度，亮度
             由于蓝色在H中间的连续部分，所以可以给色彩值设定一个上下限，保留上下限之间的像素。
       位置：https://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html
       cv函数：cv.inRange
       return: None
    """
    def nothing(x):
        pass

    def createTrackbar():
        """描述：创建轨迹条，必须在之前命名一个窗口，不然无法显示轨迹条
           return: None
        """
        cv.namedWindow('image_button', cv.WINDOW_NORMAL)
        cv.createTrackbar('H1','image_button',0,255, nothing) 
        cv.createTrackbar('S1','image_button',0,255, nothing) 
        cv.createTrackbar('V1','image_button',0,255, nothing) 
        cv.createTrackbar('H2','image_button',0,255, nothing) 
        cv.createTrackbar('S2','image_button',0,255, nothing) 
        cv.createTrackbar('V2','image_button',0,255, nothing) 

    def get_HSV_range():
        """描述：获取轨迹条的值
           return: None
        """
        lower = np.array([110,50,50])
        upper = np.array([130,255,255])

        lower[0] = cv.getTrackbarPos('H1','image_button')
        lower[1] = cv.getTrackbarPos('S1','image_button')
        lower[2] = cv.getTrackbarPos('V1','image_button')
        
        upper[0] = cv.getTrackbarPos('H2','image_button')
        upper[1] = cv.getTrackbarPos('S2','image_button')
        upper[2] = cv.getTrackbarPos('V2','image_button')

        return lower, upper

    def show_button_panel(lower, upper):
        """描述：创建一个颜色面板显示上下限的色彩，并显示轨迹条
           return: None
        """
        row = 300
        col = 600
        imgHSV = np.zeros((row, col*2,3), np.uint8)

        for i in range(3):
            imgHSV[0:row,0:col,i] = lower[i]
            imgHSV[0:row,col:col*2,i] = upper[i]

        img = cv.cvtColor(imgHSV, cv.COLOR_HSV2BGR)
        cv.imshow('image_button',img)
        cv.waitKey(5) #适当调整cv.waitKey的值，可以改善按钮的显示效果


    createTrackbar() # 创建轨迹条
    
    # url="rtsp://admin:B19911025B@169.254.101.208"
    # cap = cv.VideoCapture(url) # 开启摄像头
    cap = cv.VideoCapture(0) # 开启摄像头

    # 默认捕捉的蓝色, 在HSV空间的一个区域
    lower = np.array([110,50,50])
    upper = np.array([130,255,255])

    while(1):
        show_button_panel(lower, upper)        

        # 读取摄像头
        _, frame = cap.read()
        # Convert BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 获取创建轨迹条的HSV上下限的值
        lower, upper = get_HSV_range()
        # 保留lower, upper颜色区间的像素，其余像素置0
        mask = cv.inRange(hsv, lower, upper)
        # Bitwise-AND mask and original image
        taget = cv.bitwise_and(frame,frame, mask= mask)
        cv.imshow('frame',frame) 
        cv.imshow('mask',mask)
        cv.imshow('taget',taget)
        cv.waitKey(20) #适当调整cv.waitKey的值，可以改善按钮的显示效果

    cv.destroyAllWindows()

def main():
    # cvFilter2D()
    # cvBlur()
    # blending()
    # addALogo()
    # HSV_valueRange()
    trackingUsingHSV()

if __name__=="__main__":
    main()