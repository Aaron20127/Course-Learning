import cv2 as cv
import sys
import numpy as np
import copy
from numpy import linalg as la
import math

class LM_undistort:
    """
    功能: 相机的矫正，从拍摄的图片的坐标中得到矫正后的坐标
    描述: LM算法的回调函数，用于计算雅可比矩阵A，函数向量fi，求和代价函数F(x)
          这个函数主要是用于计算单应矩阵
    """
    def __init__(self, camera_matrix, distort, uv):
        self.fx = camera_matrix[0][0]
        self.fy = camera_matrix[1][1]
        self.u0 = camera_matrix[0][2]
        self.v0 = camera_matrix[1][2]
        self.k1 = distort[0]
        self.k2 = distort[1]
        self.p1 = distort[2]
        self.p2 = distort[3]
        self.k3 = distort[4]
        self.u1  = uv[0]
        self.v1  = uv[1]

    def updateparameters(self, x):
        self.xx = x[0]
        self.yy = x[1]

    def update(self, x0, d):
        """
        描述: 根据下降方向d更新坐标
        参数: x0: 原始坐标，narray, shape(m,)
              d: 计算得到的梯度方向，narray, shape(m,)
        返回: 新的值
        """
        x1 = x0 - d
        return x1

    def r(self):
        xx = self.xx
        yy = self.yy

        ret = (xx**2 + yy**2)**(0.5)
        return ret

    def xxx(self):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        p1 = self.p1
        p2 = self.p2        
        xx = self.xx
        yy = self.yy
        r  = self.r()

        ret = xx * (1 + k1*r**2 + k2*r**4 + k3*r**6) + \
              2*p1*xx*yy + p2*(r**2 + 2*xx**2)
        return ret

    def yyy(self):
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        p1 = self.p1
        p2 = self.p2
        xx = self.xx
        yy = self.yy
        r  = self.r()

        ret = yy * (1 + k1*r**2 + k2*r**4 + k3*r**6) + \
              2*p2*xx*yy + p1*(r**2 + 2*yy**2)
        return ret

    def u(self):
        fx = self.fx
        u0 = self.u0
        xxx = self.xxx()

        ret = fx * xxx + u0
        return ret

    def v(self):
        fy = self.fy
        v0 = self.v0
        yyy = self.yyy()

        ret = fy * yyy + v0
        return ret

    def fi(self):
        u1 = self.u1
        v1 = self.v1
        u  = self.u()
        v  = self.v()

        ret = ((u1 - u)**2 + (v1 - v)**2)**(0.5)
        return ret

    def r_xx(self):
        xx = self.xx
        r = self.r()

        ret = xx / r
        return ret

    def r_yy(self):
        yy = self.yy
        r = self.r()

        ret = yy / r
        return ret

    def xxx_xx(self):
        xx = self.xx
        yy = self.yy
        r  = self.r()
        r_xx = self.r_xx()

        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2

        ret = (1 + k1*r**2 + k2*r**4 + k3*r**6) + \
              xx*(2*k1*r + 4*k2*r**3 + 6*k3*r**5)*r_xx + \
              2*p1*yy + p2*(2*r*r_xx + 4*xx)

        return ret

    def xxx_yy(self):
        xx = self.xx
        yy = self.yy
        r  = self.r()
        r_yy = self.r_yy()

        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2

        ret = xx*(2*k1*r + 4*k2*r**3 + 6*k3*r**5)*r_yy + \
              2*p1*xx + 2*p2*r*r_yy

        return ret

    def yyy_xx(self):
        xx = self.xx
        yy = self.yy
        r  = self.r()
        r_xx = self.r_xx()

        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2

        ret = yy*(2*k1*r + 4*k2*r**3 + 6*k3*r**5)*r_xx + \
              2*p2*yy + 2*p1*r*r_xx

        return ret

    def yyy_yy(self):
        xx = self.xx
        yy = self.yy
        r  = self.r()
        r_yy = self.r_yy()

        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2

        ret = (1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r + 4*k2*r**3 + 6*k3*r**5)*r_yy + \
                2*p2*xx + p1*(2*r*r_yy + 4*yy)

        return ret

    def u_xx(self):
        xxx_xx = self.xxx_xx()
        fx = self.fx

        ret = fx*xxx_xx
        return ret

    def u_yy(self):
        xxx_yy = self.xxx_yy()
        fx = self.fx

        ret = fx*xxx_yy
        return ret

    def v_xx(self):
        yyy_xx = self.yyy_xx()
        fy = self.fy

        ret = fy*yyy_xx
        return ret

    def v_yy(self):
        yyy_yy = self.yyy_yy()
        fy = self.fy

        ret = fy*yyy_yy
        return ret

    def fi_xx(self):
        u1 = self.u1
        v1 = self.v1
        u  = self.u()
        v  = self.v()
        u_xx = self.u_xx()
        v_xx = self.v_xx()
        fi   = self.fi()

        ret = ((u1 - u)*u_xx + (v1 - v)*v_xx) / fi
        return ret

    def fi_yy(self):
        u1 = self.u1
        v1 = self.v1
        u  = self.u()
        v  = self.v()
        u_yy = self.u_yy()
        v_yy = self.v_yy()
        fi   = self.fi()

        ret = ((u1 - u)*u_yy + (v1 - v)*v_yy) / fi
        return ret

    def F(self, x):
        """
        描述: 求解代价函数fi的平方和
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 代价值
        """
        self.updateparameters(x)
        ret = self.fi()**2

        return ret

    def fiColumnVector(self, x):
        """
        描述: 求不同匹配点下的函数fi的值组成的向量
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 函数fi组成的列向量， narray, shape(m,1)
        """
        self.updateparameters(x)
        ret = np.zeros((1, 1))
        ret[0][0] = self.fi()

        return ret

    def jacobianMatrix(self, x):
        """
        描述: 函数fi向量的雅克比矩阵
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 雅可比矩阵, narray, shape(m,n)
        """
        self.updateparameters(x)
        A = np.zeros((1, x.shape[0]))

        A[0][0]  = self.fi_xx()
        A[0][1]  = self.fi_yy()

        return A


class LM_findIntrinsicAndExtrinsicParameters:
    """
    功能: 用于计算相机的内参和外参
    描述: LM算法的回调函数，用于计算雅可比矩阵A，函数向量fi，求和代价函数F(x)
          这个函数主要是用于计算单应矩阵
    """
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def updateparameters(self, x):
        self.r11, self.r12, self.r13, self.r21, self.r22, \
        self.r23, self.r31, self.r32, self.r33, self.t1, \
        self.t2, self.t3, self.k1, self.k2, self.k3, self.p1, \
        self.p2, self.fx, self.fy, self.u0, self.v0 = x

    def update(self, x0, d):
        """
        描述: 根据下降方向d更新坐标
        参数: x0: 原始坐标，narray, shape(m,)
              d: 计算得到的梯度方向，narray, shape(m,)
        返回: 新的值
        """
        x1 = x0 - d
        return x1

    def xx(self, x, s):
        X, Y, Z = s 
        r11 = self.r11
        r12 = self.r12
        r13 = self.r13
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t1 = self.t1
        t3 = self.t3

        ret = (r11*X + r12*Y + r13*Z + t1) / (r31*X + r32*Y + r33*Z + t3)
        return ret

    def yy(self, x, s):
        X, Y, Z = s 
        r21 = self.r21
        r22 = self.r22
        r23 = self.r23
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t2 = self.t2
        t3 = self.t3

        ret = (r21*X + r22*Y + r23*Z + t2) / (r31*X + r32*Y + r33*Z + t3)
        return ret

    def r(self, x, s):
        xx = self.xx(x,s)
        yy = self.yy(x,s)

        ret = (xx**2 + yy**2)**(0.5)
        return ret

    def xxx(self, x, s):
        self.updateparameters(x)
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        p1 = self.p1
        p2 = self.p2        
        xx = self.xx(x,s)
        yy = self.yy(x,s)
        r  = self.r(x,s)

        ret = xx * (1 + k1*r**2 + k2*r**4 + k3*r**6) + \
              2*p1*xx*yy + p2*(r**2 + 2*xx**2)
        return ret

    def yyy(self, x, s):
        self.updateparameters(x)
        k1 = self.k1
        k2 = self.k2
        k3 = self.k3
        p1 = self.p1
        p2 = self.p2
        xx = self.xx(x,s)
        yy = self.yy(x,s)
        r  = self.r(x,s)

        ret = yy * (1 + k1*r**2 + k2*r**4 + k3*r**6) + \
              2*p2*xx*yy + p1*(r**2 + 2*yy**2)
        return ret

    def u(self, x, s):
        self.updateparameters(x)
        fx = self.fx
        u0 = self.u0
        xxx = self.xxx(x, s)

        ret = fx * xxx + u0
        return ret

    def v(self, x, s):
        self.updateparameters(x)
        fy = self.fy
        v0 = self.v0
        yyy = self.yyy(x, s)

        ret = fy * yyy + v0
        return ret


    def fi_r11(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_r11 = X / (r31*X + r32*Y + r33*Z + t3)

        r_r11  = xx*xx_r11 / r

        xxx_r11 = xx_r11*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_r11 + 4*k2*r**3*r_r11 + 6*k3*r**5*r_r11) + \
                2*p1*yy*xx_r11 + p2*(2*r*r_r11 + 4*xx*xx_r11)

        yyy_r11 = yy*(2*k1*r*r_r11 + 4*k2*r**3*r_r11 + 6*k3*r**5*r_r11) + \
                2*p2*yy*xx_r11 + p1*(2*r*r_r11)

        u_r11 = fx*xxx_r11
        v_r11 = fy*yyy_r11

        ret = ((u1 - u)*u_r11 + (v1 - v)*v_r11) / fi
        return ret

    def fi_r12(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_r12 = Y / (r31*X + r32*Y + r33*Z + t3)

        r_r12  = xx*xx_r12 / r

        xxx_r12 = xx_r12*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_r12 + 4*k2*r**3*r_r12 + 6*k3*r**5*r_r12) + \
                2*p1*yy*xx_r12 + p2*(2*r*r_r12 + 4*xx*xx_r12)

        yyy_r12 = yy*(2*k1*r*r_r12 + 4*k2*r**3*r_r12 + 6*k3*r**5*r_r12) + \
                2*p2*yy*xx_r12 + p1*(2*r*r_r12)

        u_r12 = fx*xxx_r12
        v_r12 = fy*yyy_r12

        ret = ((u1 - u)*u_r12 + (v1 - v)*v_r12) / fi
        return ret

    def fi_r13(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_r13 = Z / (r31*X + r32*Y + r33*Z + t3)

        r_r13  = xx*xx_r13 / r

        xxx_r13 = xx_r13*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_r13 + 4*k2*r**3*r_r13 + 6*k3*r**5*r_r13) + \
                2*p1*yy*xx_r13 + p2*(2*r*r_r13 + 4*xx*xx_r13)

        yyy_r13 = yy*(2*k1*r*r_r13 + 4*k2*r**3*r_r13 + 6*k3*r**5*r_r13) + \
                2*p2*yy*xx_r13 + p1*(2*r*r_r13)

        u_r13 = fx*xxx_r13
        v_r13 = fy*yyy_r13

        ret = ((u1 - u)*u_r13 + (v1 - v)*v_r13) / fi
        return ret

    def fi_t1(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_t1 = 1 / (r31*X + r32*Y + r33*Z + t3)

        r_t1 = xx*xx_t1 / r

        xxx_t1 = xx_t1*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_t1 + 4*k2*r**3*r_t1 + 6*k3*r**5*r_t1) + \
                2*p1*yy*xx_t1 + p2*(2*r*r_t1 + 4*xx*xx_t1)

        yyy_t1 = yy*(2*k1*r*r_t1 + 4*k2*r**3*r_t1 + 6*k3*r**5*r_t1) + \
                2*p2*yy*xx_t1 + p1*(2*r*r_t1)

        u_t1 = fx*xxx_t1
        v_t1 = fy*yyy_t1

        ret = ((u1 - u)*u_t1 + (v1 - v)*v_t1) / fi
        return ret

    def fi_r21(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        yy_r21 = X / (r31*X + r32*Y + r33*Z + t3)

        r_r21 = yy*yy_r21 / r

        yyy_r21 = yy_r21*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_r21 + 4*k2*r**3*r_r21 + 6*k3*r**5*r_r21) + \
                2*p2*xx*yy_r21 + p1*(2*r*r_r21 + 4*yy*yy_r21)

        xxx_r21 = xx*(2*k1*r*r_r21 + 4*k2*r**3*r_r21 + 6*k3*r**5*r_r21) + \
                2*p1*xx*yy_r21 + p2*(2*r*r_r21)

        u_r21 = fx*xxx_r21
        v_r21 = fy*yyy_r21

        ret = ((u1 - u)*u_r21 + (v1 - v)*v_r21) / fi
        return ret

    def fi_r22(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        yy_r22 = Y / (r31*X + r32*Y + r33*Z + t3)

        r_r22 = yy*yy_r22 / r

        yyy_r22 = yy_r22*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_r22 + 4*k2*r**3*r_r22 + 6*k3*r**5*r_r22) + \
                2*p2*xx*yy_r22 + p1*(2*r*r_r22 + 4*yy*yy_r22)

        xxx_r22 = xx*(2*k1*r*r_r22 + 4*k2*r**3*r_r22 + 6*k3*r**5*r_r22) + \
                2*p1*xx*yy_r22 + p2*(2*r*r_r22)

        u_r22 = fx*xxx_r22
        v_r22 = fy*yyy_r22

        ret = ((u1 - u)*u_r22 + (v1 - v)*v_r22) / fi
        return ret

    def fi_r23(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        yy_r23 = Z / (r31*X + r32*Y + r33*Z + t3)

        r_r23 = yy*yy_r23 / r

        yyy_r23 = yy_r23*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_r23 + 4*k2*r**3*r_r23 + 6*k3*r**5*r_r23) + \
                2*p2*xx*yy_r23 + p1*(2*r*r_r23 + 4*yy*yy_r23)

        xxx_r23 = xx*(2*k1*r*r_r23 + 4*k2*r**3*r_r23 + 6*k3*r**5*r_r23) + \
                2*p1*xx*yy_r23 + p2*(2*r*r_r23)

        u_r23 = fx*xxx_r23
        v_r23 = fy*yyy_r23

        ret = ((u1 - u)*u_r23 + (v1 - v)*v_r23) / fi
        return ret

    def fi_t2(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        yy_t2 = 1 / (r31*X + r32*Y + r33*Z + t3)

        r_t2 = yy*yy_t2 / r

        yyy_t2 = yy_t2*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_t2 + 4*k2*r**3*r_t2 + 6*k3*r**5*r_t2) + \
                2*p2*xx*yy_t2 + p1*(2*r*r_t2 + 4*yy*yy_t2)

        xxx_t2 = xx*(2*k1*r*r_t2 + 4*k2*r**3*r_t2 + 6*k3*r**5*r_t2) + \
                2*p1*xx*yy_t2 + p2*(2*r*r_t2)

        u_t2 = fx*xxx_t2
        v_t2 = fy*yyy_t2

        ret = ((u1 - u)*u_t2 + (v1 - v)*v_t2) / fi
        return ret

    def fi_r31(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r11 = self.r11
        r12 = self.r12
        r13 = self.r13
        r21 = self.r21
        r22 = self.r22
        r23 = self.r23
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t1  = self.t1
        t2  = self.t2
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_r31 = -X*(r11*X + r12*Y + r13*Z + t1) / ((r31*X + r32*Y + r33*Z + t3)**2)
        yy_r31 = -X*(r21*X + r22*Y + r23*Z + t2) / ((r31*X + r32*Y + r33*Z + t3)**2)

        r_r31 = (xx*xx_r31 + yy*yy_r31) / r

        xxx_r31 = xx_r31*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_r31 + 4*k2*r**3*r_r31 + 6*k3*r**5*r_r31) + \
                2*p1*(xx*yy_r31 + yy*xx_r31)+ p2*(2*r*r_r31 + 4*xx*xx_r31)

        yyy_r31 = yy_r31*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_r31 + 4*k2*r**3*r_r31 + 6*k3*r**5*r_r31) + \
                2*p2*(xx*yy_r31 + yy*xx_r31)+ p1*(2*r*r_r31 + 4*yy*yy_r31)

        u_r31 = fx*xxx_r31
        v_r31 = fy*yyy_r31

        ret = ((u1 - u)*u_r31 + (v1 - v)*v_r31) / fi
        return ret

    def fi_r32(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r11 = self.r11
        r12 = self.r12
        r13 = self.r13
        r21 = self.r21
        r22 = self.r22
        r23 = self.r23
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t1  = self.t1
        t2  = self.t2
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_r32 = -Y*(r11*X + r12*Y + r13*Z + t1) / ((r31*X + r32*Y + r33*Z + t3)**2)
        yy_r32 = -Y*(r21*X + r22*Y + r23*Z + t2) / ((r31*X + r32*Y + r33*Z + t3)**2)

        r_r32 = (xx*xx_r32 + yy*yy_r32) / r

        xxx_r32 = xx_r32*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_r32 + 4*k2*r**3*r_r32 + 6*k3*r**5*r_r32) + \
                2*p1*(xx*yy_r32 + yy*xx_r32)+ p2*(2*r*r_r32 + 4*xx*xx_r32)

        yyy_r32 = yy_r32*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_r32 + 4*k2*r**3*r_r32 + 6*k3*r**5*r_r32) + \
                2*p2*(xx*yy_r32 + yy*xx_r32)+ p1*(2*r*r_r32 + 4*yy*yy_r32)

        u_r32 = fx*xxx_r32
        v_r32 = fy*yyy_r32

        ret = ((u1 - u)*u_r32 + (v1 - v)*v_r32) / fi
        return ret

    def fi_r33(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r11 = self.r11
        r12 = self.r12
        r13 = self.r13
        r21 = self.r21
        r22 = self.r22
        r23 = self.r23
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t1  = self.t1
        t2  = self.t2
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_r33 = -Z*(r11*X + r12*Y + r13*Z + t1) / ((r31*X + r32*Y + r33*Z + t3)**2)
        yy_r33 = -Z*(r21*X + r22*Y + r23*Z + t2) / ((r31*X + r32*Y + r33*Z + t3)**2)

        r_r33 = (xx*xx_r33 + yy*yy_r33) / r

        xxx_r33 = xx_r33*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_r33 + 4*k2*r**3*r_r33 + 6*k3*r**5*r_r33) + \
                2*p1*(xx*yy_r33 + yy*xx_r33)+ p2*(2*r*r_r33 + 4*xx*xx_r33)

        yyy_r33 = yy_r33*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_r33 + 4*k2*r**3*r_r33 + 6*k3*r**5*r_r33) + \
                2*p2*(xx*yy_r33 + yy*xx_r33)+ p1*(2*r*r_r33 + 4*yy*yy_r33)

        u_r33 = fx*xxx_r33
        v_r33 = fy*yyy_r33

        ret = ((u1 - u)*u_r33 + (v1 - v)*v_r33) / fi
        return ret

    def fi_t3(self, x, s, d):
        self.updateparameters(x)
        X, Y, Z = s
        u1 = d[0]
        v1 = d[1]
        r11 = self.r11
        r12 = self.r12
        r13 = self.r13
        r21 = self.r21
        r22 = self.r22
        r23 = self.r23
        r31 = self.r31
        r32 = self.r32
        r33 = self.r33
        t1  = self.t1
        t2  = self.t2
        t3  = self.t3
        k1  = self.k1
        k2  = self.k2
        k3  = self.k3
        p1  = self.p1
        p2  = self.p2
        fx  = self.fx
        fy  = self.fy

        xx  = self.xx(x,s)
        yy  = self.yy(x,s)
        r   = self.r(x,s)
        u   = self.u(x,s)
        v   = self.v(x,s)
        fi  = self.fi(x,s,d)

        xx_t3 = -1*(r11*X + r12*Y + r13*Z + t1) / ((r31*X + r32*Y + r33*Z + t3)**2)
        yy_t3 = -1*(r21*X + r22*Y + r23*Z + t2) / ((r31*X + r32*Y + r33*Z + t3)**2)

        r_t3 = (xx*xx_t3 + yy*yy_t3) / r

        xxx_t3 = xx_t3*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                xx*(2*k1*r*r_t3 + 4*k2*r**3*r_t3 + 6*k3*r**5*r_t3) + \
                2*p1*(xx*yy_t3 + yy*xx_t3)+ p2*(2*r*r_t3 + 4*xx*xx_t3)

        yyy_t3 = yy_t3*(1 + k1*r**2 + k2*r**4 + k3*r**6) + \
                yy*(2*k1*r*r_t3 + 4*k2*r**3*r_t3 + 6*k3*r**5*r_t3) + \
                2*p2*(xx*yy_t3 + yy*xx_t3)+ p1*(2*r*r_t3 + 4*yy*yy_t3)

        u_t3 = fx*xxx_t3
        v_t3 = fy*yyy_t3

        ret = ((u1 - u)*u_t3 + (v1 - v)*v_t3) / fi
        return ret

    def fi_k1(self, x, s, d):
        self.updateparameters(x)
        u1 = d[0]
        v1 = d[1]
        fx = self.fx
        fy = self.fy
        u  = self.u(x,s)
        v  = self.v(x,s)
        r = self.r(x,s)
        xx = self.xx(x,s)
        yy = self.yy(x,s)
        fi = self.fi(x,s,d)

        xxx_k1 = xx*r**2
        yyy_k1 = yy*r**2

        u_k1 = fx*xxx_k1
        v_k1 = fy*yyy_k1

        ret = ((u1 - u)*u_k1 + (v1 - v)*v_k1) / fi
        return ret

    def fi_k2(self, x, s, d):
        self.updateparameters(x)
        u1 = d[0]
        v1 = d[1]
        fx = self.fx
        fy = self.fy
        u  = self.u(x,s)
        v  = self.v(x,s)
        r = self.r(x,s)
        xx = self.xx(x,s)
        yy = self.yy(x,s)
        fi = self.fi(x,s,d)

        xxx_k2 = xx*r**4
        yyy_k2 = yy*r**4

        u_k2 = fx*xxx_k2
        v_k2 = fy*yyy_k2

        ret = ((u1 - u)*u_k2 + (v1 - v)*v_k2) / fi
        return ret

    def fi_k3(self, x, s, d):
        self.updateparameters(x)
        u1 = d[0]
        v1 = d[1]
        fx = self.fx
        fy = self.fy
        u  = self.u(x,s)
        v  = self.v(x,s)
        r = self.r(x,s)
        xx = self.xx(x,s)
        yy = self.yy(x,s)
        fi = self.fi(x,s,d)

        xxx_k3 = xx*r**6
        yyy_k3 = yy*r**6

        u_k3 = fx*xxx_k3
        v_k3 = fy*yyy_k3

        ret = ((u1 - u)*u_k3 + (v1 - v)*v_k3) / fi
        return ret

    def fi_p1(self, x, s, d):
        self.updateparameters(x)
        u1 = d[0]
        v1 = d[1]
        fx = self.fx
        fy = self.fy
        u  = self.u(x,s)
        v  = self.v(x,s)
        xx = self.xx(x,s)
        yy = self.yy(x,s)
        r  = self.r(x,s)
        fi = self.fi(x,s,d)

        xxx_p1 = 2*xx*yy
        yyy_p1 = r**2 + 2*yy**2

        u_p1 = fx*xxx_p1
        v_p1 = fy*yyy_p1

        ret = ((u1 - u)*u_p1 + (v1 - v)*v_p1) / fi
        return ret

    def fi_p2(self, x, s, d):
        self.updateparameters(x)
        u1 = d[0]
        v1 = d[1]
        fx = self.fx
        fy = self.fy
        u  = self.u(x,s)
        v  = self.v(x,s)
        xx = self.xx(x,s)
        yy = self.yy(x,s)
        r  = self.r(x,s)
        fi = self.fi(x,s,d)

        xxx_p2 = r**2 + 2*xx**2
        yyy_p2 = 2*xx*yy

        u_p2 = fx*xxx_p2
        v_p2 = fy*yyy_p2

        ret = ((u1 - u)*u_p2 + (v1 - v)*v_p2) / fi
        return ret

    def fi_fx(self, x, s, d):
        self.updateparameters(x)
        u1 = d[0]
        u  = self.u(x,s)
        xxx = self.xxx(x, s)
        fi = self.fi(x,s,d)

        u_fx = xxx
        ret = (u1 - u)*u_fx / fi
        return ret

    def fi_fy(self, x, s, d):
        self.updateparameters(x)
        v1 = d[1]
        yyy = self.yyy(x, s)
        v  = self.v(x,s)
        fi = self.fi(x,s,d)

        v_fy = yyy
        ret = (v1 - v)*v_fy / fi
        return ret

    def fi_u0(self, x, s, d):
        self.updateparameters(x)
        u1 = d[0]
        u  = self.u(x,s)
        fi = self.fi(x,s,d)

        u_u0 = 1
        ret = (u1 - u)*u_u0 / fi
        return ret

    def fi_v0(self, x, s, d):
        self.updateparameters(x)
        v1 = d[1]
        v  = self.v(x,s)
        fi = self.fi(x,s,d)

        v_v0 = 1
        ret = (v1 - v)*v_v0 / fi
        return ret

    def fi(self, x, s, d):
        """
        描述: 求单个函数fi的值
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
              s: 世界坐标, narray, shape(m,)
              d: 图像的像素坐标, narray, shape(m,)
        返回: 函数值
        """
        u1 = d[0]
        v1 = d[1]
        u  = self.u(x, s)
        v  = self.v(x, s) 

        ret = ((u1 - u)**2 + (v1 - v)**2)**(0.5)
        return ret

    def F(self, x):
        """
        描述: 求解代价函数fi的平均值
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 代价值
        """
        ret = 0.0
        for i in range(self.src.shape[0]):
            ret += ((self.fi(x, self.src[i], self.dst[i]))) ** 2

        return ret

    def fiColumnVector(self, x):
        """
        描述: 求不同匹配点下的函数fi的值组成的向量
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 函数fi组成的列向量， narray, shape(m,1)
        """
        ret = np.zeros((self.src.shape[0], 1))
        for i in range(self.src.shape[0]):
            ret[i][0] = self.fi(x, self.src[i], self.dst[i])

        return ret

    def jacobianMatrix(self, x):
        """
        描述: 函数fi向量的雅克比矩阵
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 雅可比矩阵, narray, shape(m,n)
        """
        A = np.zeros((self.src.shape[0], x.shape[0]))

        for i in range(A.shape[0]):
            A[i][0]  = self.fi_r11(x, self.src[i], self.dst[i])
            A[i][1]  = self.fi_r12(x, self.src[i], self.dst[i])
            A[i][2]  = self.fi_r13(x, self.src[i], self.dst[i])
            A[i][3]  = self.fi_r21(x, self.src[i], self.dst[i])
            A[i][4]  = self.fi_r22(x, self.src[i], self.dst[i])
            A[i][5]  = self.fi_r23(x, self.src[i], self.dst[i])
            A[i][6]  = self.fi_r31(x, self.src[i], self.dst[i])
            A[i][7]  = self.fi_r32(x, self.src[i], self.dst[i])
            A[i][8]  = self.fi_r33(x, self.src[i], self.dst[i])

            A[i][9]  = self.fi_t1(x, self.src[i], self.dst[i])
            A[i][10] = self.fi_t2(x, self.src[i], self.dst[i])
            A[i][11] = self.fi_t3(x, self.src[i], self.dst[i])

            A[i][12] = self.fi_k1(x, self.src[i], self.dst[i])
            A[i][13] = self.fi_k2(x, self.src[i], self.dst[i])
            A[i][14] = self.fi_k3(x, self.src[i], self.dst[i])

            A[i][15] = self.fi_p1(x, self.src[i], self.dst[i])
            A[i][16] = self.fi_p2(x, self.src[i], self.dst[i])

            A[i][17] = self.fi_fx(x, self.src[i], self.dst[i])
            A[i][18] = self.fi_fy(x, self.src[i], self.dst[i])

            A[i][19] = self.fi_u0(x, self.src[i], self.dst[i])
            A[i][20] = self.fi_v0(x, self.src[i], self.dst[i])

        return A


class LM_findHomography:
    """
    功能: 用于计算相机标定时的单应矩阵
    描述: LM算法的回调函数，用于计算雅可比矩阵A，函数向量fi，求和代价函数F(x)
          这个函数主要是用于计算单应矩阵
    """
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def update(self, x0, d):
        """
        描述: 根据下降方向d更新坐标
        参数: x0: 原始坐标，narray, shape(m,)
              d: 计算得到的梯度方向，narray, shape(m,)
        返回: 新的值
        """
        x1 = x0 + d
        x1 = 1.01*x1
        return x1

    def fi(self, x, s, d):
        """
        描述: 求单个函数fi的值
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
              s: 世界坐标, narray, shape(m,)
              d: 图像的像素坐标, narray, shape(m,)
        返回: 函数值
        """
        x1=s[0]
        y1=s[1]
        x2=d[0]
        y2=d[1]

        ret = ((x2 - (x[0]*x1 + x[1]*y1 + x[2])/(x[6]*x1 + x[7]*y1 + x[8]))**2 + \
                (y2 - (x[3]*x1 + x[4]*y1 + x[5])/(x[6]*x1 + x[7]*y1 + x[8]))**2)**(0.5)

        return ret

    def F(self, x):
        """
        描述: 求解代价函数fi的平方和
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 代价值
        """
        ret = 0.0
        for i in range(self.src.shape[0]):
            ret += ((self.fi(x, self.src[i], self.dst[i]))**2)

        return ret
    
    def fiColumnVector(self, x):
        """
        描述: 求不同匹配点下的函数fi的值组成的向量
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 函数fi组成的列向量， narray, shape(m,1)
        """
        ret = np.zeros((self.src.shape[0], 1))
        for i in range(self.src.shape[0]):
            ret[i][0] = self.fi(x, self.src[i], self.dst[i])

        return ret

    def jacobianMatrix(self, x):
        """
        描述: 函数fi向量的雅克比矩阵
        参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
        返回: 雅可比矩阵, narray, shape(m,n)
        """
        def H1(x, s, d):
            """第1列到第3列需要的参数
            """
            x1=s[0]
            y1=s[1]
            x2=d[0]
            y2=d[1]

            p1 = x2 - (x[0]*x1 + x[1]*y1 + x[2])/(x[6]*x1 + x[7]*y1 + x[8])
            p2 = x[6]*x1 + x[7]*y1 + x[8]

            return -p1/(p2*self.fi(x,s,d))

        def H2(x, s, d):
            """第4列到第6列需要的参数
            """
            x1=s[0]
            y1=s[1]
            x2=d[0]
            y2=d[1]

            p1 = y2 - (x[3]*x1 + x[4]*y1 + x[5])/(x[6]*x1 + x[7]*y1 + x[8])
            p2 = x[6]*x1 + x[7]*y1 + x[8]

            return -p1/(p2*self.fi(x,s,d))

        def H3(x, s, d):
            """第7列到第9列需要的参数
            """
            x1=s[0]
            y1=s[1]
            x2=d[0]
            y2=d[1]

            p1 = x2 - (x[0]*x1 + x[1]*y1 + x[2])/(x[6]*x1 + x[7]*y1 + x[8])
            q1 = (x[0]*x1 + x[1]*y1 + x[2])/((x[6]*x1 + x[7]*y1 + x[8])**2)

            p2 = y2 - (x[3]*x1 + x[4]*y1 + x[5])/(x[6]*x1 + x[7]*y1 + x[8])
            q2 = (x[3]*x1 + x[4]*y1 + x[5])/((x[6]*x1 + x[7]*y1 + x[8])**2)
            
            return (p1*q1 + p2*q2) / self.fi(x,s,d)


        A = np.zeros((self.src.shape[0], x.shape[0]))

        for i in range(A.shape[0]):
            A[i][0] = self.src[i][0] * H1(x, self.src[i], self.dst[i])
            A[i][1] = self.src[i][1] * H1(x, self.src[i], self.dst[i])
            A[i][2] = H1(x, self.src[i], self.dst[i])

            A[i][3] = self.src[i][0] * H2(x, self.src[i], self.dst[i])
            A[i][4] = self.src[i][1] * H2(x, self.src[i], self.dst[i])
            A[i][5] = H2(x, self.src[i], self.dst[i])

            A[i][6] = self.src[i][0] * H3(x, self.src[i], self.dst[i])
            A[i][7] = self.src[i][1] * H3(x, self.src[i], self.dst[i])
            A[i][8] = H3(x, self.src[i], self.dst[i])

        return A


class LM_testQuadraticNonlinearEquation:
    """
    功能: 测试LM算法
    描述: 使用LM算法求解二次函数fx = c1*x*x + c2的系数
    """
    def __init__(self, src):
        self.src = src

    def update(self, x0, d):
        """
        描述: 根据下降方向d更新坐标
        参数: x0: 原始坐标，narray, shape(m,)
              d: 计算得到的梯度方向，narray, shape(m,)
        返回: 新的值
        """
        x1 = x0 + d
        return x1

    def fi(self, x, s):
        """返回单个函数fi(x)的值
        """
        xi=s[0]
        yi=s[1]

        ret = yi - (x[0]*xi*xi + x[1])
        return ret

    def F(self, x):
        """返回所有函数fi(x)的平方和
        """
        ret = 0.0
        for i in range(self.src.shape[0]):
            ret = ret + ((self.fi(x, self.src[i]))**2)

        # print("F: " , ret)
        return ret
    
    def fiColumnVector(self, x):
        """返回所有函数fi(x)组成的列向量
        """
        ret = np.zeros((self.src.shape[0], 1))
        for i in range(self.src.shape[0]):
            ret[i][0] = self.fi(x, self.src[i])

        return ret

    def jacobianMatrix(self, x):
        """返回雅克比矩阵
        """
        A = np.zeros((self.src.shape[0], x.shape[0]))

        for i in range(A.shape[0]):
            A[i][0] = -self.src[i][0] ** 2
            A[i][1] = -1

        return A


def nonlinearLeastSquare_LM(x, calmat, alpha=0.01, beta=10.0, e=0.001, op=False, increment=9):
    """
    描述: Levenberg Marquardt算法主流程，参考《最优理论与算法2版》陈柏霖 326页
    参数: x: 当前单应矩阵的迭代值, narray, shape(m,)
          calmat: LM计算初始矩阵的类，用于对雅克比矩阵和代价函数等值的计算
          alpha: 初始权值，权值越大，每次迭代的下降方向越趋向于梯度的负方向
          beta: 改变alpha的系数
          e: 误差精度
          op: 是否开启打印，默认关闭
    返回: 最优参数
    """
    cal_mat=True
    A=None
    f=None
    alpha=alpha
    x0 = x
    miniFx=10000000000
    minix=x0
    iteration=0
    numFunCall=0

    if op: print("Fxinit: ", calmat.F(x))

    while(True):
        iteration+=1

        # 计算初始矩阵
        if cal_mat:
            numFunCall+=1
            alpha = alpha / beta / increment
            # alpha = alpha / beta
            f = calmat.fiColumnVector(x0)
            A = calmat.jacobianMatrix(x0)

        # alpha增大到非常大时，会出现错误，变成nan
        if math.isinf(alpha):
            alpha = 1e-306

        # 计算下降方向，若B不可逆，则增大alpha
        BB = (np.dot(A.T, A)) + (alpha*np.eye(A.shape[1]))
        # while (la.matrix_rank(BB) != BB.shape[1]):
        #     alpha = alpha*beta
        #     BB = np.dot(A.T, A) + alpha*np.eye(A.shape[1])        

        d = -1 * np.dot( la.inv(BB), np.dot(A.T, f))
        # 求单应矩阵时d前取正号
        x1 = calmat.update(x0, d.flatten())
        # 求相机参数和矫正时，d前取负号
        # x1 = x0 - d.flatten() 

        Fx1 = calmat.F(x1)
        Fx0 = calmat.F(x0)

        e1 = la.norm(np.dot(A.T,f), ord=2)

        # 打印
        if op:
            if (miniFx > Fx0 or miniFx > Fx1):
                if Fx0 > Fx1:
                    miniFx = Fx1
                    minix = x1
                else:
                    miniFx = Fx0

            print("miniFx:", miniFx,
                    "iteration:", iteration, 
                    "numFunCall:", numFunCall, 
                    "alpha:", alpha, 
                    "beta:", beta , 
                    "increment:", increment)

            # print("x0:\n", x0)

            # print("iteration: ", i)
            # print("cal_mat: ", cal_mat)
            # print("d:\n", d.flatten())
            # print("x1:\n", x1)
            # print("x0:\n", x0)
            # print("minix: \n", minix)
            # print("Fx1: ", Fx1)
            # print("Fx0: ", Fx0)
            # print("miniFx: ", miniFx)
            # print("e: ", e1)
            # print("")

        if Fx1 < Fx0:
            if e1 <= e:
                return x1
            else:
                # 下降成功，重头开始迭代
                x0 = x1
                cal_mat = True
        else:
            if e1 <= e:
                return x0
            else:
                # 下降失败，则增大alpha，朝靠近梯度负方向更新d
                alpha=beta*alpha
                cal_mat = False



if __name__=="__main__":
    # 拟合二次函数fx = a*x*x + b; 求系数a,b
    c = np.array([2,2]) #初始猜测系数
    xy = np.array([[1,2],[2,5],[3,10]]) # 拟合点
    nonlinearLeastSquare_LM(c, LM_testQuadraticNonlinearEquation(xy), op = True)