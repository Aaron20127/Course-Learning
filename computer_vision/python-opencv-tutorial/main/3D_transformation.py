"""
1.模拟相机模型，包括畸变模型k1,k2,k3,p1,p2,s1,s2.
2.模拟论文《A new calibration model of camera lens distortion》的模型
3.从网格中容易看到畸变的形状
"""

import cv2 as cv
import sys
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

sys.path.append("D:/common") 
import misc_utils
import math_utils
import opencv

def isPlaneVisible(vertex):
    """
    描述: 判断平面是否可见，通过平面的3个顶点计算出平面的法向量，
          在计算这些顶点到小孔的向量与法向量的夹角，夹角小于90则该平面可见
    参数: vertex: 每一行表示一个三维坐标，narray, shape(m,3)
    返回: True 平面可见， False 平面不可见
    """
    vector_1 = vertex[2] - vertex[0]
    vector_2 = vertex[3] - vertex[1]
    normal = np.cross(vector_1, vector_2)

    ret = True
    for v in vertex:
        #v取负号的原因是因为光线是从平面射向小孔的
        if math_utils.vectorsCosine(-v, normal) <= 0: 
            ret = False
            break
    
    return ret

def rigidTransformation_3D(input, rotation, t):
    """
    描述: 对3维坐标点做旋转平移变换，旋转先绕z,再绕y, 最后绕x旋转
    参数: input: 每一行表示一个三维坐标，narray, shape(m,3)
          rotation: 角度，弧度，list (3), list[0]=theta_x, 
                    list[1]=theta_y, list[2]=theta_z
          t: 平移向量，narray, shape(1,3)
    返回: 刚性变换后的矩阵，narray, shape(m,3)
    """
    R = np.dot(math_utils.rotation_3D_x(rotation[0]), \
        np.dot(math_utils.rotation_3D_y(rotation[1]), \
               math_utils.rotation_3D_z(rotation[2])))
    ret = np.dot(R, input.T).T + t

    return ret

def perspectiveTransformation_3D(units, translation, rotation,
                                 alpha, beta, skew, u0, v0, 
                                 k1, k2, k3, p1, p2, s1, s2, s3, s4,
                                 inverse=False):
    """
    描述: 对坐标先做刚性变换，再做透视变换
    参数: units：单元列表，lists，每个单元由字典组成，
                        {
                            "draw":是否需要绘制该面
                            "points":点集合
                            "vertex":平面多边形顶点坐标
                            "type":是平面还是线条
                        }

          translation: 先对物体平移，narray, shape(3,)
          rotation: 角度，弧度，list (3), list[0]=theta_x, 
                    list[1]=theta_y, list[2]=theta_z
          alpha: 透视变换的f/s，f(焦距)，s(CCD像素的物理长度)
          beta:  透视变换的f/l，f(焦距)，l(CCD像素的物理宽度)
          skew:  图像坐标的倾斜角，正常为90度
          u0:    图像原点x坐标
          v0:    图像原点y坐标
          inverse：生成是否倒立的像，Fasle(正立的像)，True(倒立的像)
    返回: None，新生成的坐标"new_points"直接添加到对应单元字典中
    """
    for unit in units:

        ## 1.计算每个面的法向量与平面上点到小孔的向量的夹角，夹角小于90度，则该面可见
        if unit['type'] == "polyhedron":
            vertex = rigidTransformation_3D(unit['vertex'], rotation, translation)

            if isPlaneVisible(vertex) == False:
                unit["draw"] = False
            else:
                unit["draw"] = True

        ## 2.透视变换
        if unit["draw"] == True:
            # 物体在相机坐标系下移动，方便观察
            new_points = rigidTransformation_3D(unit['points'], rotation, translation)

            # 归一化坐标，并是否倒立显示图像
            if not inverse:
                A = np.diag([1,1,-1])
                new_points = np.dot(new_points, A)

            new_points = new_points.T / new_points.T[-1] # 除上每个点的距离Z，得到二维坐标增广矩阵                

            # 畸变k1,k2,k3,p1,p2
            for c in range(new_points.shape[1]):
                x = new_points[0][c]
                y = new_points[1][c]
                rr = x**2 + y**2
                # 1.标准标定模型，径向畸变保留第一项，菱形畸变也只有s1和s31起主要作用
                new_points[0][c] = \
                    x*(1 + k1*rr + k2*rr**2 + k3*rr**3) + 2*p1*x*y + p2*(rr + 2*x**2) + s1*rr + s2*rr**2
                new_points[1][c] = \
                    y*(1 + k1*rr + k2*rr**2 + k3*rr**3) + 2*p2*x*y + p1*(rr + 2*y**2) + s3*rr + s4*rr**2

                # 2.论文《A new calibration model of camera lens distortion》中的标定模型
                ##   为了按钮方便使，用s1表示q1,s3表示q2
                # q1 = s1
                # q2 = s3
                # new_points[0][c] = \
                #     x*(1 + k1*rr + k2*rr**2) + q1*rr + x*(p1*x + p2*y)
                # new_points[1][c] = \
                #     y*(1 + k1*rr + k2*rr**2) + q2*rr + y*(p1*x + p2*y)

            # 乘上内参矩阵
            K = np.array([[alpha,  -alpha/math.tan(skew),  u0],
                          [0,       beta/math.sin(skew),   v0],
                          [0,       0,                     1]])

            uv_e = np.dot(K, new_points) # 乘上内参矩阵
            uv = uv_e[0:-1] # 删除最后一行

            unit['new_points'] = uv.T.reshape((-1,1,2)).astype("int32") #转换成cv.fillPoly 能够绘制的坐标格式
 

def polyhedronPlaneUnits():
    """
    描述: 生成多面体的平面单元列表，放入列表中，每个单元由字典组成，
            {
                "draw":是否需要绘制该面（True）
                "points":多边体平面的多边形的边的点集合，使用右手定则让点的顺序生成的法向量朝外。
                         密度自己定义。narray, shape(m,3)
                "vertex":多边体平面多边形顶点坐标。narray, shape(m,3)
            }
    参数：None
    返回：生成多面体的平面单元列表
    """
    ## 1. 多面体每个面的顶点，顺序按右手定则法向量朝外方向排列，
    # 多面体（正六面体）的中心是(0.5,0.5,0.5)，先让坐标全部为正，
    # 否则np.mgrid可能出错
    vertexes = np.array([[[0,0,0], [0,1,0], [1,1,0], [1,0,0]],
                         [[0,0,1], [1,0,1], [1,1,1], [0,1,1]],
                         [[0,0,0], [1,0,0], [1,0,1], [0,0,1]],
                         [[0,1,0], [0,1,1], [1,1,1], [1,1,0]],
                         [[0,0,0], [0,0,1], [0,1,1], [0,1,0]],
                         [[1,0,0], [1,1,0], [1,1,1], [1,0,1]]])              

    ## 2. 每个面的边上生成一圈坐标点
    transvector = np.array([[-0.5, -0.5, -0.5]])  # 平移多面体使相机坐标系中心在立方体质心
    units = []
    num = 10 # 两点之间插入点的个数
    for i in range(vertexes.shape[0]):
        points = None
        for j in range(vertexes.shape[1]):
            a = None
            b = None

            if j != 3:
                a = vertexes[i][j]
                b = vertexes[i][j+1]
            else:
                a = vertexes[i][j]
                b = vertexes[i][0]

            c = opencv.insertPointsBwteen2Points(a,b,num)

            if j == 0:
                points = c
            else:
                points = np.vstack((points, c))
        

        vertex = vertexes[i] + transvector # 所有点要平移到质心
        points += transvector
        unit = {
                "type": "polyhedron",
                "vertex": vertex, # 每个面的顶点
                "points": points, # 面的边上的点
                "draw": True 
        }
        units.append(unit)

    return  units

def lineUnits(scale, num):
    units = []

    p1 = np.array([-1,1,0])*scale
    p2 = np.array([1,1,0])*scale
    p3 = np.array([1,-1,0])*scale
    p4 = np.array([-1,-1,0])*scale

    line_ver_p1 = opencv.insertPointsBwteen2Points(p1, p2, num)
    line_ver_p2 = opencv.insertPointsBwteen2Points(p4, p3, num)

    for i in range(line_ver_p1.shape[0]):
        line_points = opencv.insertPointsBwteen2Points(line_ver_p1[i,:], line_ver_p2[i,:], num)
        unit = {
                "type": "line",
                "points": line_points, 
                "draw": True 
        }
        units.append(unit)

    line_ver_p1 = opencv.insertPointsBwteen2Points(p1, p4, num)
    line_ver_p2 = opencv.insertPointsBwteen2Points(p2, p3, num)

    for i in range(line_ver_p1.shape[0]):
        line_points = opencv.insertPointsBwteen2Points(line_ver_p1[i,:], line_ver_p2[i,:], num)
        unit = {
                "type": "line",
                "points": line_points, 
                "draw": True 
        }
        units.append(unit)

    return units

def drawLine(img, units):
    """
    描述: 绘制线条
    参数：img：画布
          units：单元
    返回：None
    """
    color_plane = [(0,255,255), (255,0,255), (255,255,0), \
                   (0,0,255), (0,255,0), (255,0,0)]

    for i in range(len(units)):
        unit = units[i]
        if unit['draw'] and unit['type'] == 'line':
            cv.polylines(img, [unit['new_points']], False, color_plane[0], thickness=1)

def drawPolyhedron(img, units):
    """
    描述: 根据平面单元绘制多边形透视后的平面，被遮挡的平面不会被绘制
    参数：img：画布
          plane_units：多面体平面多边形单元裂变
    返回：None
    """
    color_plane = [(0,255,255), (255,0,255), (255,255,0), \
                   (0,0,255), (0,255,0), (255,0,0)]

    for i in range(len(units)):
        unit = units[i]
        if unit['draw'] and unit['type'] == 'polyhedron':
            cv.fillPoly(img,[unit['new_points']],color_plane[i])

def testPerspectiveTransformation_3D():
    """
    描述: 1.使用透视变换绘制立方体，整个模型使用相机透视矩阵即K(3x3内参矩阵)，M(3x4)外参矩阵。
          2.相机坐标和世界坐标相同
          3.使用了斜交skew，畸变k1和k2,p1,p2
    参数: None
    返回: None
    """
    img_canvas = np.zeros((720,700,3), np.uint8) 

    # 1.创建按钮
    button_units = [['tx', 400,   800],
                    ['ty', 400,   800],
                    ['tz', 0,   400],
                    ['rx', 0,   1000],
                    ['ry', 0,   1000],
                    ['rz', 0,   1000],
                    ['alpha', 400, 2000],
                    ['beta', 400, 2000],
                    ['skew', 0, 90],
                    ['u0', 360,   800],
                    ['v0', 300,   800],
                    ['k1', 500,   1000],
                    # ['k2', 300,   600],
                    # ['k3', 300,   600],
                    ['p1', 300,   600],
                    ['p2', 300,   600],
                    ['s1', 300,   600],
                    # ['s2', 300,   600],
                    # ['s4', 300,   600],
                    ['s3', 300,   600]]

    track_bar = opencv.createTrackbar(button_units, "trackbar")

    ## 2.创建多面体的面单元
    polyhedron_units = polyhedronPlaneUnits()
    ## 3.插入一些便于观察畸变的线条
    line_units = lineUnits(10, 30)

    while(1):
        cv.imshow('img_canvas',img_canvas)
        cv.waitKey(1)

        # 更新画布
        img_canvas = np.zeros((620,700,3), np.uint8) 

        ## 3.透视变换
        [tx, ty, tz, rx, ry, rz, alpha, beta, skew, u0, v0, k1, p1, p2, s1, s3] =\
             track_bar.getTrackbarPos()

        # 调节参数，因为trackbar只能用正整数造成的限制
        translation = np.array([(tx-400)/100,(ty-400)/100,(tz-200)/10])
        rotation = ((rx/1000)*2*math.pi, (ry/1000)*2*math.pi, (rz/1000)*2*math.pi)
        skew = (90-skew)/90 * math.pi/2
        k1 = (k1-500) / 100
        # k2 = (k2-300) / 100
        # k3 = (k3-300) / 100
        p1 = (p1-300) / 100
        p2 = (p2-300) / 100
        s1 = (s1-300) / 100
        # s2 = (s2-300) / 100
        s3 = (s3-300) / 100
        # s4 = (s4-300) / 100

        perspectiveTransformation_3D( polyhedron_units, translation, rotation,
                                    alpha, beta, skew, u0, v0, 
                                    k1, 0, 0, p1, p2, s1, 0, s3, 0,
                                    inverse=False)
        
        perspectiveTransformation_3D(line_units, translation, rotation,
                                alpha, beta, skew, u0, v0, 
                                k1, 0, 0, p1, p2, s1, 0, s3, 0,
                                inverse=False)
        ## 3.绘制图形平面
        drawLine(img_canvas,  line_units)
        drawPolyhedron(img_canvas,  polyhedron_units)

        ## 4.打印输出
        opencv.putText(img_canvas, '  pos:  %.2f  %.2f  %.2f' % \
                (translation[0],translation[1],translation[2]), (10, 60))
        opencv.putText(img_canvas, '  rot:  %.2f  %.2f  %.2f' % \
                (rotation[0]/math.pi*180, rotation[1]/math.pi*180, rotation[2]/math.pi*180), (10, 80))
        opencv.putText(img_canvas, 'alpha:  %.2f' % (alpha), (10, 100))
        opencv.putText(img_canvas, ' beta:  %.2f' % (beta), (10, 120))
        opencv.putText(img_canvas, ' skew:  %.2f' % (skew/math.pi*180), (10, 140))
        opencv.putText(img_canvas, '   u0:  %.2f' % (u0), (10, 160))
        opencv.putText(img_canvas, '   v0:  %.2f' % (v0), (10, 180))        
        opencv.putText(img_canvas, '   k1:  %.2f' % (k1), (10, 200))
        # opencv.putText(img_canvas, '   k2:  %.2f' % (k2), (10, 220))
        # opencv.putText(img_canvas, '   k3:  %.2f' % (k3), (10, 240))
        opencv.putText(img_canvas, '   p1:  %.2f' % (p1), (10, 260))
        opencv.putText(img_canvas, '   p2:  %.2f' % (p2), (10, 280))
        opencv.putText(img_canvas, '   s1:  %.2f' % (s1), (10, 300))
        opencv.putText(img_canvas, '   s3:  %.2f' % (s3), (10, 320))

    cv.destroyAllWindows()


testPerspectiveTransformation_3D()

