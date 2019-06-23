"""
学习贝叶斯分类
使用的正态分布下的极大似然估计求解条件概率密度
类别 = (先验概率1 * 类条件概率1) > (先验概率2 * 类条件概率2)
"""

import scipy.io as sio
import numpy as np
import cv2 
import copy
import sys
import os

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)

def imageMask(img, mask, new_elem = [255,255,255]):
    """给图片中不需要的像素变成同一种颜色
        img: 图像矩阵，narray，shape(n,m)或shape(n,m,r)
        mask: 屏蔽位，narray，shape(n,m)
        new_elem: 新的像素值
        return: 新的图片矩阵，narray，shape(n,m)或shape(n,m,r)
    """
    new_img = copy.deepcopy(img)
    [row, col] = mask.shape

    for i in range(row):
        for j in range(col):
            if mask[i][j] == 0:
                new_img[i][j] = new_elem

    return new_img

# 正态极大似然估计
def getMean(x):
    """求正态分布期望的最大似然估计
        x: 训练样本值，narray，shape(n,m)，每行为一个样本，每一列为一类特征
        return: 均值向量，shape(1,m)
    """
    n = len(x)
    return np.sum(x, axis=0) * 1.0 / n

def getVariance(x, u):
    """求正态分布协方差矩阵的最大似然估计
        x: 训练样本值，narray，shape(n,m)，每行为一个样本，每一列为一类特征
        u: 估计的样本均值，narray，shape(n,)，非矩阵形式
        return：协方差矩阵，narray，shape(n,n)，n为一个样本特征的维数
    """
    n = len(x)
    deta = x - u
    return 1.0 * np.dot(deta.T, deta) / n

def classConditionalProbability(u, cov_mat, x):
    """正态分布函数的概率分布，类条件分布
        x: 测试样本值，narray，shape(n,m)，每行为一个样本，每一列为一类特征
        u: 估计的样本均值，narray，shape(n,)，非矩阵形式
        cov_mat: 估计的样本的协方差矩阵，narray，shape(n,n)
        return：测试样本的在该类中的概率，narray，shape(n,)
    """
    cla_con_pro = []
    d = len(u) # x的维数
    cov_mat_deter = np.linalg.det(cov_mat) # 协方差矩阵的行列式
    cov_mat_inv = np.linalg.inv(cov_mat) # 协方差矩阵的逆

    for xi in x:
        deta = xi - u # 样本离期望的差值
        pro = np.exp(-0.5 * np.dot(deta, np.dot(cov_mat_inv, deta.T))) /\
              ((2.0*np.pi)**(d/2.0) * cov_mat_deter**(0.5))
        cla_con_pro.append(pro)

    return np.array(cla_con_pro)

def BayesianDecision(training_sample, lable, test_img):
    """ 根据训练样本和标签得到类条件概率，再通过贝叶斯公式对测试图片的像素进行分类
        最后返回分类后的二值图
        training_sample: 训练样本，narray，shape(n,m)
        lable: 标签，narray，shape(n,)
        test_img: 测试图片矩阵，narray，shape(n,m)或shape(n,m,r)
        return: 二值图，0或255，narray，shape(n,m)
    """
    ## 预处理测试数据，使每个像素列对齐，并使值在0-1之间
    row = test_img.shape[0]
    col = test_img.shape[1]
    test_sample = (test_img / 255.0).reshape(row * col, training_sample.shape[1])

    ## 1.样本分类
    class_1 = []
    class_2 = []

    for i in range(len(lable)):
        if lable[i] == 1:
            class_1.append(training_sample[i])
        else:
            class_2.append(training_sample[i])

    class_1 = np.array(class_1)
    class_2 = np.array(class_2)

    ## 2.计算灰度均值和协方差矩阵
    mean_1 = getMean(class_1)
    mean_2 = getMean(class_2)

    cov_mat_1 = getVariance(class_1, mean_1)
    cov_mat_2 = getVariance(class_2, mean_2)

    ## 3.正太分布下贝叶斯决策
    # 先验概率
    n_1 = len(class_1)
    n_2 = len(class_2)
    prior_pro_1 = 1.0 * n_1 / (n_1 + n_2) # 一类的先验概率
    prior_pro_2= 1.0 - prior_pro_1 # 二类的先验概率

    # 类条件概率
    class_con_pro_1 = \
        classConditionalProbability(mean_1, cov_mat_1, test_sample)
    class_con_pro_2 = \
        classConditionalProbability(mean_2, cov_mat_2, test_sample)

    # 贝叶斯决策
    result = ((prior_pro_1 * class_con_pro_1) < \
            (prior_pro_2 * class_con_pro_2))

    # 逻辑矩阵还原成二值图，0或255
    img_binary = (result + 0) * 255
    img_binary = (img_binary.astype('uint8')).reshape(row, col) # 还原成原图片矩阵

    return img_binary

def main():
    """颜色分类
    """
    Mask = sio.loadmat('tmp/Mask.mat')['Mask']
    training_data = sio.loadmat('tmp/array_sample.mat')['array_sample']
    lable = training_data[:,[4]].flatten() # 标签

    ### 1. 测试灰度图
    img_gray = cv2.imread("tmp/309.bmp", 0)  # 得到灰度图
    img_gray = imageMask(img_gray, Mask, new_elem=255) # 屏蔽背景
    training_sample_gray = training_data[:,[0]]    # 获取灰度训练样本
    img_binary_gray = BayesianDecision(training_sample_gray, lable, img_gray) # 贝叶斯决策

    ### 2. 测试rgb图
    img_rgb = cv2.imread("tmp/309.bmp", 1)   # 得到BGR图
    img_rgb = imageMask(img_rgb, Mask) # 屏蔽背景
    training_sample_rgb = training_data[:,[3,2,1]]   # 获取BGR训练样本，注意opencv颜色通道的顺序BGR
    img_binary_rgb = BayesianDecision(training_sample_rgb, lable, img_rgb) # 贝叶斯决策

    # 显示图片
    cv2.imshow("img_gray", img_gray) 
    cv2.imshow("img_binary_gray", img_binary_gray) 
    cv2.imshow("img_rgb", img_rgb) 
    cv2.imshow("img_binary_rgb", img_binary_rgb) 

    cv2.imwrite("img_gray.jpg", img_gray)
    cv2.imwrite("img_binary_gray.jpg", img_binary_gray)
    cv2.imwrite("img_rgb.jpg", img_rgb)
    cv2.imwrite("img_binary_rgb.jpg", img_binary_rgb)

    cv2.waitKey (0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()