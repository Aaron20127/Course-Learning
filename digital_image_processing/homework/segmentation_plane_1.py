
import cv2
import os
import sys

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)

img_path = r'pictures/plane.bmp'
img = cv2.imread(img_path)
cv2.imshow('ori',img)

# 转换灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

# 用高斯滤波处理原图像降噪
blur = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow('blur',blur)

# canny 边缘检测
canny = cv2.Canny(blur, 30, 110)

cv2.imshow('canny',canny)

# 图像形态学
# 定义不同的核 MORPH_RECT（方形核）， MORPH_CROSS（十字核）， MORPH_ELLIPSE（椭圆核）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 12))
# 膨胀和腐蚀
dilated = cv2.dilate(canny, kernel)

cv2.imshow('dilated',dilated)

erode = cv2.erode(canny, kernel)

cv2.imshow('erode',erode)
# 获取差分图
result = cv2.absdiff(dilated, erode)
cv2.imshow('absdiff',result)

# 模糊图像二值化
ret, thresh = cv2.threshold(result, 100,255,cv2.THRESH_OTSU)

# 闭运算
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

canny_2 = cv2.Canny(closed, 30, 110) 

cv2.imshow('Division', closed)
cv2.imshow('Contours', canny_2)

cv2.waitKey()
cv2.destroyAllWindows()
