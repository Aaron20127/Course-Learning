"""
一些计算中用到的基本函数
"""

import numpy as np
from numpy import linalg as la
import math


def rotation_3D_x(theta):
    """
    描述: 获取坐标绕x轴旋转的3维旋转矩阵
    参数: theta: 旋转的角度，弧度作单位
    返回: 3维旋转矩阵，narray, shape(3,3)
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[1,0,0],[0, cos, -sin],[0, sin, cos]])

def rotation_3D_y(theta):
    """
    描述: 获取绕y轴旋转的3维旋转矩阵
    参数: theta: 旋转的角度，弧度作单位
    返回: 3维旋转矩阵，narray, shape(3,3)
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[cos,0,sin],[0, 1, 0],[-sin, 0, cos]])


def rotation_3D_z(theta):
    """
    描述: 获取绕z轴旋转的3维旋转矩阵
    参数: theta: 旋转的角度，弧度作单位
    返回: 3维旋转矩阵，narray, shape(3,3)
    """
    cos = math.cos(theta)
    sin = math.sin(theta)
    return np.array([[cos,-sin,0],[sin, cos, 0],[0, 0, 1]])


def generate_sample(min, max, size = (10,2)):
    """
    描述: 随机产生一个元素为浮点型的矩阵
    参数: min: 元素的最小值, flaot
          max: 元素的最大值, float
          size: 矩阵的维度，m*n
    返回: 随机矩阵，narray, shape(m,n)
    """
    return ((max-min) * np.random.random_sample(size) + min)


def vectorsCosine(a, b):
    """
    描述: 计算两个向量的夹角的余弦值，两个向量可以是任意维数
    参数: a: 第一个向量, list (3)
          b: 第二个向量, list (3)
    返回: 向量的夹角的余弦值
    """
    return np.sum(a*b) / (np.linalg.norm(a) * np.linalg.norm(b))


def fullRankDecomposition(m):
    """
    描述: 矩阵最大秩分解，原理是将奇异值分解中的大于0的奇异值和对应的向量保留下来。
          得到U1,L,V1，然后B=U1，D=L*U1
    参数: m: 需要分解的矩阵, narray, shape(m,n)
    返回: B: narray, shape(m,r)
          D: narray, shape(r,n)
    """
    # m = np.eye(4)
    # m = np.array([[2,3,2],[2,3,1],[2,3,1]]).T
    # print(m)
    U, sigma, VT=la.svd(m)

    r=0
    for i in range(sigma.shape[0]):
        if (sigma.shape[0]-1 == i):
            r=i+1 # 当取整个数组时，要加个1才能取整个数组
        if (sigma[i] == 0): 
            r=i
            break
    
    U1 = U[:, 0:r]
    L = np.diag(sigma[0:r])
    V1 = VT[0:r]

    B = U1
    D = np.dot(L, V1)

    # print(np.dot(B,D))
    return B, D


def generalizedInverseMatrixAplus(m):
    """
    描述: 通过最大秩分解求广义逆矩阵A+,A+ = D.T*((D*D.T).-1)*((B.T*B).-1)*B.T
    参数: m: 需要求广义逆的矩阵，narray, shape(m,n)
    返回: 广义逆矩阵A+
    """
    B, D = fullRankDecomposition(m)
    A1 = np.dot(D.T, la.inv(np.dot(D,D.T)))
    A2 = np.dot(la.inv(np.dot(B.T,B)), B.T)
    Aplus = np.dot(A1, A2)

    return Aplus


def optimalApproximationSolution(A, b):
    """
    描述: 求矩阵最佳逼近解，即拟合最好的解
    参数: A:系数矩阵，narray, shape(m,n)
          b:非齐次方程的值的列向量，narray, shape(m,1)
    返回: x: 最佳拟合解，narray, shape(n,1)
    """
    Aplus = generalizedInverseMatrixAplus(A)
    x = np.dot(Aplus, b)
    # print(x)
    return x


if __name__=="__main__":
    c = np.array([[0,1,1]]).T
    theta = math.pi/2
    n_c = np.dot(rotation_3D_x(theta), c)
    print("x:\n", c.T, '\n', n_c.T)

    c = np.array([[1,0,1]]).T
    theta = math.pi/2
    n_c = np.dot(rotation_3D_y(theta), c)
    print("\ny:\n", c.T, '\n', n_c.T)

    c = np.array([[1,1,0]]).T
    theta = math.pi/2
    n_c = np.dot(rotation_3D_z(theta), c)
    print("\nz:\n", c.T, '\n', n_c.T)