import cv2 as cv
import numpy as np

def normlizeImage2Uint8(img):
    """
    描述: 将图像归一化到uint8，主要是为了方便显示图片，
          虽然能使用全部数据，但不一定比截取的图片效果好。
    参数: img: narray, shape(m,n) 或者 shape(m,n,3)
    返回: 归一化后的图片，narray
    """
    img_new = np.float64(img)
    img_norm = (img_new - img_new.min()) / (img_new.max() - img_new.min())
    return np.uint8(255 * img_norm)

def drawHist(img, size=(300,256*2+1)):
    """
    描述: 绘制灰度直方图到图片上
    参数: img: 需要绘制直方图分布的灰度图，narray, shape(m,n)
          size: 生成直方图的大小，size[0]图像高度，size[1]图像宽度
    返回: None
    """
    color_plane = [(0,255,255), (255,0,255), (255,255,0), \
                   (0,0,255), (0,255,0), (255,0,0)]

    # 计算直方图，并归一化
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    hist = hist/hist.max() # 这个归一化是为了直方图在显示的时候，使最大值占满屏幕

    img_hist = np.ones((size[0],size[1],3), np.uint8) * 255
    x_interval = (size[1]-1)//256 # 条形图的间隔

    for i in range((hist.shape[0])):

        # 逆时针旋转画条形
        j = i*x_interval
        x0 = j
        y0 = size[0]-1

        x1 = j + x_interval
        y1 = y0

        x2 = x1
        y2 = size[0] - 1 - (hist[i][0]*(size[0]-1))

        x3 = x0
        y3 = y2
        
        bar = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]], np.int32)
        bar = bar.reshape((-1,1,2))

        cv.fillPoly(img_hist,[bar],color_plane[4])

    # 将直方图迁移到图像中间
    top = 30
    bottom = 30
    right = 30
    left = 30
    img_hist_ajust = np.ones((size[0] + top + bottom, \
                        size[1] + right + left, 3), np.uint8) * 255
    img_hist_ajust[top:size[0]+top, left:size[1]+left] = img_hist
    
    return img_hist_ajust

def putText(img, text, pos, fontScale=0.5):
    """
    描述: 绘制字符到图片上
    参数: img: 图片矩阵，narray, shape(m,n,3)
          text: 字符串名称，string
          text_pos: 字符串的位置
          fontScale: 字符串大小，float
    返回: None
    """
    cv.putText(img, text, pos, cv.FONT_HERSHEY_SIMPLEX,\
               fontScale, (255,255,255), 1, cv.LINE_AA)

def insertPointsBwteen2Points(a, b, num):
    """
    描述: 在任意维的两个坐标中插入num个点，共生成num+1个点
    参数: a：起点，narray, shape(n,)
          b: 终点，narray, shape(n,)
          num: a与b之间需要插入的点的个数
    返回: 生成的坐标矩阵，narray, shape(num+1,n)
    """
    dem = a.shape[0]
    ret = a.reshape(1,dem)
    de = b-a
    seg = num+1
    for i in range(1,seg):
        new = a + i/seg * de
        ret = np.vstack((ret, new.reshape(1,dem)))
    
    ret = np.vstack((ret, b.reshape(1,dem)))

    return ret

class createTrackbar():
    """
    描述: 创建任意多个拖动条
    """
    def __init__(self, button_units, bar_name, size=(200, 400), block=False):
        """
        描述: 初始化拖动条
        参数: button_units: list, 每个元素是一个list(3)， 
                            list[0] 滑动条名称，
                            list[1] 初始值（正整数），
                            list[2] 最大值（正整数）

              bar_name: 整个滑动条图片的名称
              size:     附加在按钮上的图片大小
              block:    阻塞机制，如果参数没变化就一直等待,每次循环阻塞默认10ms
        """
        self._button_units = button_units
        self._bar_name = bar_name
        self._size = size
        self._block = block
        self._paralist = []
        
        imgTrackbar = np.zeros((size[0],size[1],3), np.uint8) #大的黑板，保证拖动条能完全显示
        cv.namedWindow(bar_name, 0)

        for uint in button_units:
            cv.createTrackbar(uint[0], bar_name, uint[1], uint[2], self.nothing) 

        cv.imshow(bar_name,imgTrackbar)

    def nothing(self,x):
        """
        描述：控件回调函数
        返回: None
        """
        pass

    def getTrackbarPos(self):
        """
        描述: 读取所有滑动条的参数
        返回：None
        """
        paralist = []
        for unit in self._button_units:
            paralist.append(cv.getTrackbarPos(unit[0], self._bar_name))

        ## 如果参数没变化就等待
        if self._block:
            while(paralist == self._paralist):
                paralist = []
                for unit in self._button_units:
                    paralist.append(cv.getTrackbarPos(unit[0], self._bar_name))
                cv.waitKey(10)

        self._paralist = paralist
        return paralist

class cameraUnistort():
    """
    畸变图片矫正
    """
    def __init__(self, mtx, dist, image_size):
        """
        描述: 初始化，获取矫正畸变的重映射矩阵
        参数：mtx：内参矩阵，narray, shape(3,3)
             dist：畸变系数矩阵，narray, shape(1,5), [k1, k2, p1, p2, k3]
             image_size: 图像长和宽，注意应该是(image.shape[1], image.shape[0])
        返回：矫正后的图片矩阵，narray, shape(m,n)
        """
        ## 计算新相机矩阵和没有黑边的区域roi
        newcameramtx, roi = cv.getOptimalNewCameraMatrix( \
            mtx, dist, image_size, 0, image_size)

        ## 得到新的映射矩阵，mapx, mapy, narray, shape(m,n)
        self.mapx, self.mapy = \
            cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, image_size, 5)
        
    def undistort(self, img):
        """
        描述: 将畸变图片像素位置映射到矫正后的像素位置
        输入：img：畸变图片矩阵，narray, shape(m,n)
        返回：矫正后的图片矩阵，narray, shape(m,n)
        """
        return cv.remap(img, self.mapx, self.mapy, cv.INTER_LINEAR)

if __name__=="__main__":
    img = drawHist(np.array([[1]]))
    cv.imshow("hist", img)
    cv.waitKey(0)