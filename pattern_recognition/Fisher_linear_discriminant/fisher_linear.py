"""
学习Fisher线性判别函数
线性分类：y(x) = WT * X + W0
注意：在推导WT时，WT的正负不会影响决策面的值，而W0的正负与决策面有关
WT = Sw.I * (m1 - m2)
w0 = -1/2 * (WT*m1 + WT*m2)

Note: to run the code, you need to download the common library
"""

import os
import sys
import time

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)

import misc_utils 
import math_utils
import numpy as np
import matplotlib.pyplot as plt


def plot_data(x1, x2, w, w0, title):
    """绘制训练数据和决策面
        x1，x2: 测试样本，每一行一个样本值，narray, shape(m,n)
        w: 决策面垂直方向的方向向量，narray，shape(n,1)
        w0: 决策面的偏移量，constant
        title: 图形名称
        return：none
    """

    #1.决策面
    # 确定画线的范围
    xlength = 0 
    if len(x1): 
        xlength = np.max(np.abs(x1))

    if len(x2):
        temp = np.max(np.abs(x2))
        if temp > xlength: 
            xlength = temp

    x_fn = np.linspace(-xlength, xlength, 10, endpoint=False)
    # y_fn是令g(x) = WT * x + w0 = 0 计算出来的
    y_fn = ((-w[0] * x_fn - w0) / w[1])[0]    

    x_coordinate = [x_fn]
    y_coordinate = [y_fn]
    p_type = ['line']
    line_lable = ['decision surface']

    #2.分类点
    if len(x1):
        x_coordinate.append(x1[:,0])
        y_coordinate.append(x1[:,1])
        p_type.append("scatter")
        line_lable.append('class 1')

    if len(x2):
        x_coordinate.append(x2[:,0])
        y_coordinate.append(x2[:,1])
        p_type.append("scatter")
        line_lable.append('class 2')

    misc_utils.plot().plot_base(
        x_coordinate = x_coordinate, 
        y_coordinate = y_coordinate,
        line_lable = line_lable,
        title = title,
        x_lable = 'y',
        y_lable = 'x',
        p_type = p_type,
        axis_equal = True) 

def training_data(x1, x2, figure = False):
    """
    描述: 训练训练数据，并绘制决策面和训练数据
    参数: x1，x2: 测试样本，每一行一个样本值，narray, shape(m,n)
          figure: 是否将决策面和训练数据绘制出来
    返回: w：投影方向，narray, shape(n,1)
          w0: 决策面位移, constant
    """
    ## 1.训练
    # 每一类样本的数量
    n1 = len(x1)
    n2 = len(x2)

    # 样本均值
    m1 = 1.0/n1 * np.dot(np.ones((1,n1)), x1)
    m2 = 1.0/n2 * np.dot(np.ones((1,n2)), x2)

    # 类类离散度矩阵
    S1 = np.dot((x1 - m1).T, (x1 - m1))
    S2 = np.dot((x2 - m2).T, (x2 - m2))
    Sw = S1 + S2

    # 类内离散度矩阵的逆矩阵
    I_Sw = np.linalg.inv(Sw)

    ## 2.绘制决策面和训练数据
    if figure:
        ## 改变w为相反的方向，并不影响决策面的位置
        ## 以下是正确的决策面
        w = np.dot(I_Sw, (m1-m2).T)
        w0 = -1.0/2 * np.dot(w.T, (m1+m2).T)
        plot_data(x1, x2, w, w0, 'Fisher, training, correct, +w, +w0')

        """
        w = - np.dot(I_Sw, (m1-m2).T)
        w0 = -1.0/2 * np.dot(w.T, (m1+m2).T)
        plot_data(x1, x2, w, w0, 'Fisher, training, correct, -w, +w0')

        ## 但是改变w0会影响决策面的位置，默认w0 = -1/2 * WT * (m1+m2)
        ## 由于w0的符号反了，得到了错误的决策面
        w = np.dot(I_Sw, (m1-m2).T)
        w0 = 1.0/2 * np.dot(w.T, (m1+m2).T)
        plot_data(x1, x2, w, w0, 'Fisher, training, error, w, -w0')

        w = - np.dot(I_Sw, (m1-m2).T)
        w0 = 1.0/2 * np.dot(w.T, (m1+m2).T)
        plot_data(x1, x2, w, w0, 'Fisher, training, error, -w, -w0')
        """

    ## 3. 返回正确的方向w和偏移w0
    w = np.dot(I_Sw, (m1-m2).T)
    w0 = -1.0/2 * np.dot(w.T, (m1+m2).T)
    return w, w0

def test_data(test, w, w0, figure = True):
    """
    描述: 测试样本，并绘制数据
    参数: test: 测试样本，每一行一个样本值，narray, shape(m,n)
          w：投影方向，narray, shape(n,1)
          w0: 决策面位移, constant
          figure: 是否将决策面和训练数据绘制出来
    返回: cla_1:第一类，narray, shape(m,n)
          cla_2:第二类，narray, shape(m,n)
    """
    cla_1 = []
    cla_2 = []

    ## 分类
    gx = (np.dot(w.T, test.T) + w0) > 0
    cla = gx.flatten()

    for i in range(len(cla)):
        if cla[i]:
            cla_1.append(test[i])
        else:
            cla_2.append(test[i])
    
    cla_1 = np.array(cla_1)
    cla_2 = np.array(cla_2)

    ## 绘图
    if figure:
        plot_data(cla_1, cla_2, w, w0, 'Fisher, test')

    return cla_1, cla_2

def main():
    ## 1.训练样本,随机生成
    training_1 = math_utils.generate_sample(-5, 0, (10,2))
    training_2 = math_utils.generate_sample(0, 5, (10,2))
    w, w0 = training_data(training_1, training_2, figure=True)

    ## 2.测试样本，随机生成
    test = math_utils.generate_sample(-5, 5, (100,2))
    test_data(test, w, w0, figure=True)

    plt.show()

if __name__=="__main__":
    main()
