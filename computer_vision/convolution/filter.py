"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def cv2Filter2D():
    """描述：cv2.filter2D是通用卷积滤波函数，测试时使用方形卷积核，对彩色图像使用归一和非归一化滤波
       注意：1. cv2.filter2D是correlation而非convolution，convolution需要将kernel绕中心旋转180度
               具体参考：https://www.mathworks.com/help/images/what-is-image-filtering-in-the-spatial-domain.html#f16-20755
            2. 核的最大为11*11时使用的直接方法计算，大于11*11则使用其他方法计算
            3. 在图像边缘的地方使用核时，由于某些地方不存在像素，要使用插值方法
            4. 具体使用>>> help(cv2.filter2D)命令查看使用方法
       归一化：滤波后的像素值是周围像素的平均值，这样不会改变图像的亮度（不知道对不对），只会模糊图片。
              核中权重值越大，图像变得越明亮，反之，图像越暗（也不知道对不对）。
       
    """
    def linearFilter(img, kernel):
        dst = cv2.filter2D(img,-1,kernel)

        plt.figure()
        plt.subplot(121),plt.imshow(img),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(dst),plt.title\
            ('Averaging %dx%d' % (kernel.shape[0], kernel.shape[1]))
        plt.xticks([]), plt.yticks([])
    
    path = 'tmp/'

    ## 1. 模糊整个苹果
    img_BGR = cv2.imread(path + 'apple.jpg')
    img_RGB = img_BGR[...,::-1] # 将BGR通道转换成RGB通道
    kernel = np.ones((9,9),np.float32)/81
    linearFilter(img_RGB, kernel)

    ## 2. 矩形内的像素相同，卷积核为一个小框，相当于模糊矩形的边缘
    #     对图片放大可以发现矩形边缘已经模糊
    img_BGR = cv2.imread(path + 'squre.png')
    img_RGB = img_BGR[...,::-1]
    kernel = np.ones((9,9),np.float32)/81  # 归一化滤波，即核内所有像素的均值
    linearFilter(img_RGB, kernel)

    img_BGR = cv2.imread(path + 'squre.png')
    img_RGB = img_BGR[...,::-1]
    kernel = np.ones((9,9),np.float32)/100 # 像素的值小于归一化，即降低图像的亮度
    linearFilter(img_RGB, kernel)

    img_BGR = cv2.imread(path + 'squre.png')
    img_RGB = img_BGR[...,::-1]
    kernel = np.ones((9,9),np.float32)/60 # 像素的值大于归一化，即增加图像的亮度
    linearFilter(img_RGB, kernel)

    plt.show()

def cvBlur():
    """描述：直接使用归一化线性卷积核模糊图片，与cv2.filter2D的归一化效果相同，
            不过使用cv2.blur模糊图片更简单，因为只需要定义核的长和宽。
    """
    path = 'tmp/'
    img_BGR = cv2.imread(path + 'apple.jpg')
    img_RGB = img_BGR[...,::-1]
    blur = cv2.blur(img_RGB,(9,9))

    plt.subplot(121),plt.imshow(img_RGB),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

def main():
    # cv2Filter2D()
    cvBlur()

if __name__=="__main__":
    main()