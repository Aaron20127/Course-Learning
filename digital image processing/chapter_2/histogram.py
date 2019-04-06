
import cv2 as cv
import sys
import numpy as np
import copy
import math
from matplotlib import pyplot as plt
from matplotlib import ticker


sys.path.append("D:/common") 
import misc_utils
import math_utils
import opencv


def enhanceImageClarity():
    """
    描述: 1.通过图像相减获取血管。
          2.取反可以达到图像增强效果。
          3.使用自适应灰度直方图均衡化，可以提高图像对比度和细节
    参数: None
    返回: None
    """
    img0 = cv.imread('pictures/subtraction_0.tif')
    img1 = cv.imread('pictures/subtraction_1.tif')
    img_gray_0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
    img_gray_1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    # 1.图像相减
    img_diff = cv.subtract(img_gray_0, img_gray_1)
    img_gray_2 = img_diff - img_diff.min() # 使最小值为正

    # 2.取反增强图像效果
    img_gray_3 = 255 - img_gray_2

    # 3.创建按钮
    button_units = [['clipLimit', 8, 100],
                    ['row', 8, 200],
                    ['col', 8, 200]]

    track_bar = opencv.createTrackbar(button_units, "trackbar")

    while(1):
        cv.waitKey(1)

        # 获取trackbar参数
        [clipLimit, row, col] = track_bar.getTrackbarPos()

        # 自适应灰度调节
        # 在OpenCV中 tiles的大小默认为8×8。然后我们再分别对每一小块进行直方图均衡化。
        # 所以在每一个的区域中， 直方图会集中在某一个小的区域中（除非有噪声干扰）。
        # 如果有噪声的话，噪声会被放大。 为了避免这种情况的出现要使用对比度限制。 
        # 对于每个小块来说，如果直方图中的 bin 超过对比度的上限的话， 
        # 就把其中的像素点均匀分散到其他 bins 中， 然后在进行直方图均衡化。
        # 最后，为了去除每一个小块之间“人造的”（由于算法造成）边界， 再使用双线性差值，对小块进行缝合。
        clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(row,col))
        img_gray_createCLAHE_2 = clahe.apply(img_gray_2)
        img_gray_createCLAHE_3 = clahe.apply(img_gray_3)

        cv.imshow('img_gray_createCLAHE_2',img_gray_createCLAHE_2)
        cv.imshow('img_gray_createCLAHE_3',img_gray_createCLAHE_3)

        # 显示灰度图
        img_hist_2 = opencv.drawHist(img_gray_createCLAHE_2)
        img_hist_3 = opencv.drawHist(img_gray_createCLAHE_3)
        cv.imshow('hist_2',img_hist_2)
        cv.imshow('hist_3',img_hist_3)

def enhanceBackgroundBrightness():
    """
    描述: 1.使用cv.equalizeHist函数，会使直方图分布太均匀，
           从而使图像的对比度过高，导致细节丢失，比如说导致本例中的头部的信息丢失。
          2.cv.createCLAHE函数是先将图像分成若干块，在每一块中进行直方图均衡化。
            这样可以逐渐调节直方图的分布，动态地观察直方图的分布变化给图像带来的变化。
            且加入噪声抑制方法，这里不是太懂？？
          3.不知道cv.createCLAHE是否是我们常用的显示器调节对比度的方法？？
    参数: None
    返回: None
    """
    img = cv.imread('pictures/histogram_0.jpg', 0)

    # 直接直方图均衡化
    cv.imshow('img',img)
    img_Hist = opencv.drawHist(img)
    cv.imshow('img_Hist',img_Hist)

    img_equalizeHist = cv.equalizeHist(img)
    cv.imshow('img_equalizeHist',img_equalizeHist)
    equalizeHist = opencv.drawHist(img_equalizeHist)
    cv.imshow('equalizeHist',equalizeHist)

    # 1.创建按钮
    button_units = [['clipLimit', 1, 100],
                    ['row', 8, 200],
                    ['col', 8, 200]]

    track_bar = opencv.createTrackbar(button_units, "trackbar")

    while(1):
        cv.waitKey(1)

        # 回去按钮参数
        [clipLimit, row, col] = track_bar.getTrackbarPos()

        # 自适应灰度调节
        clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(row,col))
        img_createCLAHE = clahe.apply(img)
        cv.imshow('img_createCLAHE',img_createCLAHE)

        # 显示灰度图
        hist = opencv.drawHist(img_createCLAHE)
        cv.imshow('createCLAHE_Hist',hist)


def enhanceAllBrightness():
    """
    描述: 1.使用c = 255/img.max()乘上图片的方法获得图像的最大亮度
          2.在增大调节系数c时，可以发现图像的直方图整体是在网255级移动
    参数: None
    返回: None
    """
    img = cv.imread('pictures/house.jpg', 0)

    # 直接直方图均衡化
    cv.imshow('img',img)
    img_Hist = opencv.drawHist(img)
    cv.imshow('img_Hist',img_Hist)

    # 1.创建按钮
    button_units = [['coefficient', 210, 255]]

    track_bar = opencv.createTrackbar(button_units, "trackbar")

    while(1):
        cv.waitKey(1)

        # 获取按钮参数
        [coefficient] = track_bar.getTrackbarPos()

        # 调节亮度
        img_new = (img * (coefficient / img.max())).astype("uint8")
        cv.imshow("img_new", img_new)

        # 显示灰度图
        hist = opencv.drawHist(img_new)
        cv.imshow('img_new_Hist',hist)


def BGR_Histogram2D():
    """
    描述: 显示BGR图像的任意两个通道的2D直方图
    参数: None
    返回: None
    参考: https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/
    """
    def show_image_histogram_2d(image, bins=32, tick_spacing=5):
        """
        描述: 显示BGR图像的任意两个通道的2D直方图
        参数: image: BGR3通道图像，narray, shape(m,n,3)
            bins: 分成的显示级别，1-256级别，类似于灰度分级，级别越高统计越详细
            tick_spacing：显示的坐标间隔
        返回: None
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        channels_mapping = {0: 'B', 1: 'G', 2: 'R'}
        for i, channels in enumerate([[0, 1], [0, 2], [1, 2]]):
            hist = cv.calcHist(
                [image], channels, None, [bins] * 2, [0, 256] * 2) # [x]*2相当于将列表中的元素复制成两个

            channel_x = channels_mapping[channels[0]]
            channel_y = channels_mapping[channels[1]]

            ax = axes[i]
            ax.set_xlim([0, bins - 1])
            ax.set_ylim([0, bins - 1])

            ax.set_xlabel(f'Channel {channel_x}')
            ax.set_ylabel(f'Channel {channel_y}')
            ax.set_title(f'2D Color Histogram for {channel_x} and {channel_y}')

            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

            im = ax.imshow(hist)

        fig.colorbar(im, ax=axes.ravel().tolist(), orientation='orizontal')
        fig.suptitle(f'2D Color Histograms with {bins} bins', fontsize=16)

    img_light = cv.imread('pictures/light-tones.jpg')
    cv.imshow("img_light", img_light)
    show_image_histogram_2d(img_light, bins=32)
    plt.show()

def HSI_Histogram2D():
    """
    描述: 显示HSI图像模型的HS通道的2D直方图
    参数: None
    返回: None
    """
    def show_histogram_2d(image):
        """
        描述: 显示HSI模型的前两个通道值的2D直方图
        参数: image: HSI通道图像，narray, shape(m,n,3)
        返回: None
        参考: https://matplotlib.org/gallery/axes_grid1/simple_colorbar.html#sphx-glr-gallery-axes-grid1-simple-colorbar-py
              https://matplotlib.org/gallery/axes_grid1/demo_colorbar_with_axes_divider.html
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # 创建1*1子图
        fig, axes = plt.subplots(nrows=1, ncols=1)

        axes.set_title('2D HSI Color Histogram')
        axes.set_xlabel('S')
        axes.set_ylabel('H')

        # 行为y坐标，列为x坐标
        hist = cv.calcHist([image], [0,1], None, [180, 256], [0, 180, 0, 256])

        # 绘制条2D和彩色条
        im = axes.imshow(hist)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)   


    img_light = cv.imread('pictures/light-tones.jpg')
    cv.imshow("img_light", img_light)
    img_HSV = cv.cvtColor(img_light,cv.COLOR_BGR2HSV)
    show_histogram_2d(img_HSV)
    plt.show()

if __name__=="__main__":
    # enhanceImageClarity()
    # enhanceBackgroundBrightness()
    # enhanceAllBrightness()
    # BGR_Histogram2D()
    HSI_Histogram2D()


    

