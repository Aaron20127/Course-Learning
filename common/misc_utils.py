#!/usr/bin/python
# -*- coding: UTF8 -*-

"""
各种各样会常用的工具
"""

import time
import threading
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import cv2

import math
import pickle as cpickle #python3.5用的pickle代替cPickle
import gzip
import os.path
import os
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import multiprocessing

def write_list_to_file(file, list):
    """
    描述：将字典转换成json字符串，并写入相应文件中
    参数：file：文件路径
          list：需要保存的数据，dict
    返回：none
    """
    obj_string = json.dumps(list)
    fo = open(file, "w")
    fo.write(obj_string)
    fo.close()
    return obj_string

def read_list_from_file(file):
    """
    描述：从文件中读取字符串，并将其成json对象，即列表或字典
    参数：file：文件路径
    返回：json对象
    """
    fo = open(file, "r")
    obj = json.loads(fo.read())
    fo.close()
    return obj


#### 画图测试 https://blog.csdn.net/qq_31192383/article/details/53977822
class plot:
    def plot_base(self, y_coordinate, x_coordinate = [], line_lable = [], 
                line_color = [], title = '', x_lable = '', y_lable = '',
                x_limit = [], y_limit = [], y_scale = 'linear', p_type = [],
                grad = False, axis_equal = False):
        """
        描述：画一幅坐标曲线图，可以同时有多条曲线
        参数：y_coordinate （y坐标值，二元列表，例如[[1,2,3],[4,5,6]]，表示有两条曲线，每条曲线的y坐标为[1,2,3]和[4,5,6]）
                x_coordinate  (x坐标值，同y坐标值，如果不提供x坐标值，则默认是从0开始增加的整数)
                line_lable   （每条曲线代表的意义，就是曲线的名称，没有定义则使用默认的）
                line_color    (曲线的颜色，一维列表，如果比曲线的条数少，则循环使用给定的颜色；不给定时，使用默认颜色；
                            更多色彩查看 http://www.114la.com/other/rgb.htm)
                title        （整个图片的名称）
                x_lable      （x轴的含义）
                y_lable       (y轴的含义)
                x_limit       (x坐标的显示范围)
                y_scale       (y轴的单位比例，'linear'常规，'log'对数)
                p_type        (类型：line线条，scatter散点)
                grad          (网格)
        """

        if (x_coordinate and (len(y_coordinate) != len(x_coordinate))):
            print ("error：x坐标和y坐标不匹配！")
            sys.exit()
        
        if (line_lable and  (len(y_coordinate) != len(line_lable))):
            print ("error：线条数和线条名称数不匹配，线条数%d，线条名称数%d！" % \
                    (len(y_coordinate),len(line_lable)))     
            sys.exit()

        if not line_color:
            line_color = ['#9932CC', '#FF4040' , '#FFA933', '#CDCD00',
                            '#CD8500', '#C0FF3E', '#B8860B', '#AB82FF']
            # print "info: 未指定色彩，使用默认颜色！"

        if len(y_coordinate) > len(line_color):
            print ("warning: 指定颜色种类少于线条数，线条%d种，颜色%d种！" % \
                    (len(y_coordinate),len(line_color)))

        # plt.figure(figsize=(70, 35)) 
        plt.figure() 
        ax = plt.subplot(111)

        # 如果没有给x的坐标，设置从0开始计数的整数坐标
        if not x_coordinate:
            x_coordinate = [range(len(y)) for y in y_coordinate]

        # 如果没有给线条名称，则使用默认线条名称
        if not line_lable:
            line_lable = ["line " + str(i) for i in range(len(y_coordinate))]

        # 如果没有指定图形类型，默认画线条line
        if not p_type:
            p_type = ["line" for y in y_coordinate]

        for i in range(len(y_coordinate)):
            if p_type[i] == 'line':
                ax.plot(x_coordinate[i], y_coordinate[i], color = line_color[i%len(line_color)], \
                        linewidth = 2.0, label = line_lable[i])      
            elif p_type[i] == 'scatter': 
                ax.scatter(x_coordinate[i], y_coordinate[i],  s = 90, c=line_color[i%len(line_color)],\
                            linewidth = 2.0, alpha=0.6, marker='+', label = line_lable[i])
            else:
                print ("error：Invalid p_type %s！" % (p_type[i]))
                sys.exit()

        ax.set_title(title) # 标题
        ax.set_xlabel(x_lable) # x坐标的意义
        ax.set_ylabel(y_lable) # y坐标的意义
        ax.set_yscale(y_scale) # 'linear','log'
        ### 自适应轴的范围效果更好
        if x_limit: ax.set_xlim(x_limit) # x坐标显示的范围
        if y_limit: ax.set_ylim(y_limit) # y坐标显示范围
        
        if axis_equal: plt.axis("equal") # 横坐标和纵坐标的单位长度相同

        # plt.xticks()
        # plt.yticks()
        plt.legend(loc="best") # 线条的名称显示在右下角
        if grad: plt.grid(True) # 网格

        # plt.savefig("file.png", dpi = 200)  #保存图片，默认png     
        # plt.show()

    def plot_base_3d(self, x_coordinate, y_coordinate, z_function, title = '',
                x_lable = '', y_lable = '', z_lable = '',
                x_limit = [], y_limit = [], z_limit = []):
        """绘制3D网格图
        """
        figure = plt.figure() 
        ax = Axes3D(figure)

        #网格化数据
        X, Y = np.meshgrid(x_coordinate, y_coordinate)
        Z = z_function(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(x_lable, fontsize=14) # x坐标的意义
        ax.set_ylabel(y_lable, fontsize=14) # y坐标的意义
        ax.set_zlabel(z_lable, fontsize=14) # z坐标的意义
        if x_limit: ax.set_xlim(x_limit) # x坐标显示的范围
        if y_limit: ax.set_ylim(y_limit) # y坐标显示的范围
        if z_limit: ax.set_zlim(z_limit) # z坐标显示的范围
    
    def picture_color_maps(self):
        """绘图的plt.imshow(img, cmap=None)的所有的cmap可选的值，即绘图的色域，
           可用在test_3中
        """
        # Have colormaps separated into categories:
        # http://matplotlib.org/examples/color/colormaps_reference.html
        cmaps = [('Perceptually Uniform Sequential', [
                    'viridis', 'plasma', 'inferno', 'magma']),
                ('Sequential', [
                    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
                ('Sequential (2)', [
                    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                    'hot', 'afmhot', 'gist_heat', 'copper']),
                ('Diverging', [
                    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                    'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
                ('Qualitative', [
                    'Pastel1', 'Pastel2', 'Paired', 'Accent',
                    'Dark2', 'Set1', 'Set2', 'Set3',
                    'tab10', 'tab20', 'tab20b', 'tab20c']),
                ('Miscellaneous', [
                    'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                    'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                    'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

        nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        def plot_color_gradients(cmap_category, cmap_list, nrows):
            fig, axes = plt.subplots(nrows=nrows)
            fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
            axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

            for ax, name in zip(axes, cmap_list):
                ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
                pos = list(ax.get_position().bounds)
                x_text = pos[0] - 0.01
                y_text = pos[1] + pos[3]/2.
                fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

            # Turn off *all* ticks & spines, not just the ones with colormaps.
            for ax in axes:
                ax.set_axis_off()

        for cmap_category, cmap_list in cmaps:
            plot_color_gradients(cmap_category, cmap_list, nrows)

    def plot_picture(self, matrix, cmap=None, title=None, axis=True):
        """绘制矩阵图片
            matrix 是列表，每个元素代表一个图片的像素矩阵
            title  是列表，每个元素代表标题
            cmap   是色彩
        """
        def get_subplot_region_edge(num):
            for i in range(10000):
                if num <= i*i: 
                    return i

        total = len(matrix)
        edge = get_subplot_region_edge(total)
        plt.figure() 

        for i in range(total):
            ax = plt.subplot(edge, edge, i+1)  
            if title:
                ax.set_title(title[i], fontsize=14)

            if cmap:
                plt.imshow(matrix[i], cmap=cmap)
            else:
                plt.imshow(matrix[i])

            if not axis:
                plt.xticks([]) # 关闭图片刻度，必须放在imshow之后才生效
                plt.yticks([])

    def plot_multiple_picture(self, size, data, equal=False):
        """描述：在一幅图中简单地绘制n幅图片
        size: 元组，size[0]表示行数，size[1]表示列数 
        data: 绘图需要的信息，data是元组，每个元素也是一个元组
                data[0][0]:表示RGB矩阵, narray
                data[0][1]:表示名字，string
                data[0][2]:表示显示时的色彩，string
        return: None
        """
        count = size[0] * size[1]
        data_len = len(data)

        if data_len > count:
            print ("error：too many pictures！")
            sys.exit()

        plt.figure()
        pos = size[0]*100 + size[1]*10

        for i in range(data_len):
            plt.subplot(pos + i + 1)

            if len(data[i]) == 2:
                plt.imshow(data[i][0])
            elif len(data[i]) == 3:
                if data[i][2] == "gray":
                    plt.imshow(data[i][0], 'gray')
                else:
                    plt.imshow(data[i][0], data[i][2])

            plt.title(data[i][1])
            plt.xticks([]), plt.yticks([])
            if equal: plt.axis("equal")

    def test(self):
        ### 1. 基本曲线图
        self.plot_base([[1,2,3],[6,5,6]],
                  line_lable = ['feature 1', 'feature 2'],
                  line_color = ['#9932CC', '#FFA933'],
                  title = 'Classification',
                  x_lable = 'Epoch',
                  y_lable = 'accuracy') 

        ### 2. 3D网格图
        def fun(x, y):
            return np.sin(3*x) + np.cos(3*y)

        self.plot_base_3d(np.arange(1, 10, 0.1), np.arange(1, 10, 0.1), fun,
                    x_lable='X', y_lable='Y', z_lable='Z')
        
        ### 3. 绘制图片的颜色域
        self.picture_color_maps()

        ### 4. 图片的读取和绘制
        """读取png图片，以不同色彩画出
           绘制随机灰度图
           绘制随机二值图
        """
        np.random.seed(19680801)
        img=mpimg.imread('tmp/stinkbug.png', format='png')
        # print img

        """ 1.因为原图是灰度图，即二维图，没有RGBA颜色分配，所以默认分配的颜色为viridis，
            每一个矩阵的值代表该颜色的深浅，所以给二维图分配不同的色彩可以得到不同颜色的效果图
            2.若原图是彩色图，即三位图，则自动生成原色彩的图片，不会失真
        """ 
        ### 绿色虫子(默认cmap = viridis)
        plt.figure()
        plt.imshow(img)

        ### 火红色虫子
        self.plot_picture([img], 'hot', title=['hot'])

        ### 灰色虫子
        self.plot_picture([img], 'gray', title=['gray'])

        ### 随机产生灰度图
        img = np.random.random((28, 28)) # 返回[0.0, 0.1)之间的随机数数组
        self.plot_picture([img], 'gray', title=['random gray']) 

        ### 二值图，灰色值只取0或1
        img = np.random.randint(low=0, high=2, size = (28,28)) # 随机生成0-1之间的整数
        self.plot_picture([img], 'binary', title=['random binary']) 

        ### 5. 一幅图中绘制n幅图片
        """在同一个图片中显示多福图片
        """
        img_bgr = cv2.imread('tmp/flower.jpg', 1)   #得到BGR图
        (img_B,img_G,img_R) = cv2.split(img_bgr) #提取R、G、B分量

        data = [[img_bgr[...,::-1], 'Original'],
                [img_R, 'img_R', 'gray'],
                [img_G, 'img_G', 'gray'],
                [img_B, 'img_B', 'gray'],]

        self.plot_multiple_picture((2,2), data)

        plt.show()


if __name__=="__main__":        	
    plot().test()
