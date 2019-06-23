"""
神经网络二分类
训练样本：800张19*19图片，共两类，-1和1分别表示两类
测试样本：400张19*19图片，共两类，-1和1分别表示两类

最基本的神经网络结构，《深度学习与神经网络》的源码：
python2.7: https://github.com/mnielsen/neural-networks-and-deep-learning
python3.5: https://github.com/MichalDanielDobrzanski/DeepLearningPython35
"""

import sys
import os

# gets the absolute path to the file
abspath = os.path.abspath(os.path.dirname(__file__))
# add the library file path
sys.path.append(abspath + "/../../common")
# change the work path
os.chdir(abspath)


import misc_utils
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import copy
import network

def createVectorized(dim, j, column=True):
    """
    描述: 生成一个单位向量，第j个值为1，可以是行向量，也可以是列向量
    参数: dim：向量的维度
          j: 第几个值为1
          column: 是否是列向量，False表示行向量
    返回: 单位向量
    """
    e = 0
    if column:
        e = np.zeros((dim, 1))
    else:
        e = np.zeros((1, dim))

    e[j] = 1.0
    return e

def load_data():
    """
    描述: 从二分类样本中读取数据，并生成神经网络训练时所需要的数组格式
    参数: None
    返回: training_data：800*1的列表，其中每个元素由两个array组成，
          第一个array是361*1的样本输入，第二个array是2*1的样本标签，
          第几行的值为1，就是第几列样本。
          test_data:400*1的列表，结构和training_data相同，不同之处
          是标签是0或1的数字，0表示第一类，1表示第二类。
    注意：python3使用zip操作后的数据，必须使用list()操作才能变成元组使用
    """
    filepath = 'tmp/'
    training_input_raw = sio.loadmat(filepath + 'train_data.mat')['train_data']
    training_label_raw = sio.loadmat(filepath + 'train_label.mat')['train_label']
    test_input_raw = sio.loadmat(filepath + 'test_data.mat')['test_data']
    test_label_raw = sio.loadmat(filepath + 'test_label.mat')['test_label']

    ## 组织训练数据，将源标签1和-1改成0和1
    training_input = [np.reshape(x, (19*19, 1)) for x in training_input_raw]
    training_label = []
    for lable in training_label_raw:
        if lable == -1:
            training_label.append(createVectorized(2, 0))
        else:
            training_label.append(createVectorized(2, 1))
    training_data = zip(training_input, training_label)

    ## 组织测试数据
    test_input = [np.reshape(x, (19*19, 1)) for x in test_input_raw]
    test_label = []
    for label in test_label_raw:
        if label == -1:
            test_label.append(0)
        else:
            test_label.append(1)

    test_data = zip(test_input, test_label)
    return training_data, test_data

def showDataImage(train_num=(0,36), test_num=(0,36)):
    """
    描述: 用图片的形式显示训练样本和测试样本
    参数: train_num: 元组，显示训练数据中第train_num[0]到第train_num[1]g个图片
          test_num: 元组，显示训练数据中第test_num[0]到第test_num[1]个图片
    返回: None
    """
    def plot(data):
        imgs = []
        for img in data:
            imgs.append(img[0].reshape((19,19)).T)
        misc_utils.plot().plot_picture(imgs, 'gray', title=None) 

    training_data, test_data = load_data()
    plot(list(training_data)[train_num[0]:train_num[1]])
    plot(list(test_data)[test_num[0]:test_num[1]])

    plt.show()

def main():
    """
    描述: 用最基本的反向传播方法训练样本，代价函数使用最基本的二次代价函数，
          初试权重和偏置使用归一化方法。
    参数: None
    返回: None
    思路：使用不同的小批量和步长训练，每次训练迭代最多5000次，若在5000次之
         前测试正确率稳定在0.5%之间连续50次，则停止训练，保存数据。正常运行
         完成，则保留最后一次数据。
    """
    batchs = [2,4,5,8,10,20,25,40,80,100,200,400]
    aphas = np.linspace(0.001, 1.0, num=1000, endpoint=False)
    save_data = []

    for batch in batchs:
        for apha in aphas:
            training_data, test_data = load_data()
            net = network.Network([19*19, 30, 30, 30, 30, 2])
            test_accuracy, training_accuracy, epoch = net.SGD(training_data, 5000, batch, apha,
                               test_data=test_data, stop=(50, 0.5))
            save_data.append([float('%.2f' % test_accuracy), float('%.2f' % training_accuracy),
                             batch, float('%.2f' % apha), epoch])
            misc_utils.write_list_to_file('training_result_%d.net' % batch, save_data)

if __name__=="__main__":
    # 以图片形式显示训练数据和测试数据
    # showDataImage((300,325),(300,325))
    main()