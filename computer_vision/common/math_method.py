
"""
对一些在计算机视觉中需要使用的数学知识进行测试
"""

import numpy as np

def SVD_test():
    """对矩阵做奇异值分解，A = U * E * VT
       U是m*m维，E是m*n维，VT是n*n维。
       la.svd(A)算出的奇异值sigma是一维向量，需要把其值按顺序放到E的对角上。
       参考：1.《矩阵理论》奇异值分解
            2.https://www.bilibili.com/video/av15971352
    """
    A=np.mat([[1,2,3],[4,5,6]])
    from numpy import linalg as la
    U,sigma,VT=la.svd(A)

    print("\nU:\n", U)
    print("\nSigma:\n", sigma)
    print("\nVT:\n", VT)

if __name__=="__main__":
    SVD_test()
