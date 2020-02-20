import numpy as np
from math import sin, cos


def find_T_matrix(pts, t_pts):
	# pts是原始坐标, t_pts是要变换过去的坐标
	A = np.zeros((8, 9))
	# 下面的循环是构建矩阵A, 然后对A进行奇异值分解
	# A = USV, V的最后一行向量就是方程的解, 也就是要求解的矩阵h
	for i in range(0, 4):
		xi = pts[:, i]  # 获得一组坐标(x y 1), 列向量
		xil = t_pts[:, i]
		xi = xi.T  # (x y 1)
		A[i*2,   3:6] = -xil[2]*xi
		A[i*2,   6: ] =  xil[1]*xi
		A[i*2+1,  :3] =  xil[2]*xi
		A[i*2+1, 6: ] = -xil[0]*xi
	# 奇异值分解, 分解后的v的最后一行向量就是解
	[U, S, V] = np.linalg.svd(A)
	H = V[-1, :].reshape((3, 3))
	return H


def getRectPts(tlx, tly, brx, bry):
	return np.matrix([[tlx, brx, brx, tlx],
					  [tly, tly, bry, bry],
					  [1., 1., 1., 1.]], dtype=float)


# 手动求透视变换的矩阵（find_T_matrix函数所需要寻找的就是这个矩阵）
def perspective_transform(wh, angles=np.array([0., 0., 0.]), zcop=1000., dpp=1000.):
	# angles to radians, 将角度转换为弧度, 这里的angles有三个值, 分别是x, y, z轴三个方向的旋转角度
	rads = np.deg2rad(angles)
	a = rads[0]
	# 绕x轴旋转的透视矩阵
	Rx = np.matrix([[1, 0, 0],
					[0, cos(a), sin(a)],
					[0, -sin(a), cos(a)]])
	a = rads[1]
	# 绕y轴旋转的透视矩阵
	Ry = np.matrix([[cos(a), 0, -sin(a)],
					[0, 1, 0],
					[sin(a), 0, cos(a)]])
	a = rads[2]
	# 绕z轴旋转的透视矩阵
	Rz = np.matrix([[cos(a), sin(a), 0],
					[-sin(a), cos(a), 0],
					[0, 0, 1]])
	R = Rx * Ry * Rz
	(w, h) = tuple(wh)
	# np.matrix是np.array的一个小分支, 其只能是二维的, 如果要做矩阵相乘, 只需要a * b, np.array做矩阵相乘需要使用np.dot()函数
	xyz = np.matrix([[0, 0, w, w],
					 [0, h, 0, h],
					 [0, 0, 0, 0]])
	hxy = np.matrix([[0, 0, w, w],
					 [0, h, 0, h],
					 [1, 1, 1, 1]])
	xyz = xyz - np.matrix([[w], [h], [0]])/2.
	xyz = R*xyz
	xyz = xyz - np.matrix([[0], [0], [zcop]])
	hxyz = np.concatenate([xyz, np.ones((1, 4))])
	P = np.matrix([[1, 0, 0, 0],
				   [0, 1, 0, 0],
				   [0, 0, -1./dpp, 0]])
	_hxy = P * hxyz
	_hxy = _hxy / _hxy[2, :]
	_hxy = _hxy + np.matrix([[w], [h], [0]]) / 2.
	return find_T_matrix(hxy, _hxy)
