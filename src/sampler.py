import cv2
import numpy as np
import random
from src.utils import im2single, getWH, hsv_transform, IOU_centre_and_dims
from src.label import Label
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts


def labels2output_map(label, lppts, dim, stride):
	# 因为输入图片是(208, 208, 3)，因此，经过4次池化之后的feature map会缩小16倍
	# dim=208
	# lppts：车牌区域的四个点的精确的坐标，一个2*4的矩阵，分别是左上，右上，右下，左下
	side = ((float(dim) + 40.)/2.)/stride  # 7.75 when dim = 208 and stride = 16
	outsize = int(dim/stride)  # 208 / 16 = 13
	Y = np.zeros((outsize, outsize, 2*4+1), dtype='float32')
	MN = np.array([outsize, outsize])  # 就是一个数组，[13, 13]
	WH = np.array([dim, dim], dtype=float)  # 数组[208, 208]
	# 因为车牌的坐标是除了高和宽的，因此车牌的坐标在对应的缩小后的feature map里面可以直接乘以缩小后的feature map高和宽
	# 得到对应的图片大小
	tlx, tly = np.floor(np.maximum(label.tl(), 0.)*MN).astype(int).tolist()
	brx, bry = np.ceil(np.minimum(label.br(), 1.)*MN).astype(int).tolist()
	# 下面的遍历是在左上和右下这两个点组成的矩形框内选取一个点
	for x in range(tlx, brx):
		for y in range(tly, bry):
			mn = np.array([float(x) + .5, float(y) + .5])
			iou = IOU_centre_and_dims(mn/MN, label.wh(), label.cc(), label.wh())
			if iou > .5:
				p_WH = lppts * WH.reshape((2, 1))  # 将四个点的坐标乘以高和宽变为真正的坐标
				p_MN = p_WH/stride   # 坐标除以步长得到feature map缩小后对应的坐标位置
				p_MN_center_mn = p_MN - mn.reshape((2, 1))  # 论文公式(3)
				p_side = p_MN_center_mn/side  # 论文alpha的，
				Y[y, x, 0] = 1.
				Y[y, x, 1:] = p_side.T.flatten()
	return Y


# 让pts变成(3, 4)的矩阵，第三行为全1的一行向量
def pts2ptsh(pts):
	return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))


def project(I, T, pts, dim):
	ptsh = np.matrix(np.concatenate((pts, np.ones((1, 4))), 0))
	ptsh = np.matmul(T, ptsh)
	ptsh = ptsh/ptsh[2]
	ptsret = ptsh[:2]
	ptsret = ptsret/dim
	Iroi = cv2.warpPerspective(I, T, (dim, dim), borderValue=.0, flags=cv2.INTER_LINEAR)
	return Iroi, ptsret


def flip_image_and_pts(I, pts):
	I = cv2.flip(I, 1)
	pts[0] = 1. - pts[0]
	idx = [1, 0, 3, 2]
	pts = pts[..., idx]
	return I, pts


def augment_sample(I, pts, dim):
	# dim = 208, I是图像(h, w, c), pts车牌位置归一化后的坐标
	maxsum, maxangle = 120, np.array([80., 80., 45.])
	angles = np.random.rand(3) * maxangle
	if angles.sum() > maxsum:
		angles = (angles/angles.sum()) * (maxangle/maxangle.sum())
	I = im2single(I)  # 车牌坐标归一化
	iwh = getWH(I.shape)  # 得到图像的width 和 height
	whratio = random.uniform(2., 4.)  # 宽高比例，只要知道高，就可以根据这个比例得到宽
	wsiz = random.uniform(dim * .2, dim * 1.)  # 生成的数据在[dim * .2, dim * 1]之间
	hsiz = wsiz/whratio
	dx = random.uniform(0., dim - wsiz)
	dy = random.uniform(0., dim - hsiz)
	pph = getRectPts(dx, dy, dx+wsiz, dy+hsiz)
	# pts是一个2*4的矩阵，iwh是2*1的宽和高矩阵，下面是求出坐标点的原始坐标
	pts = pts * iwh.reshape((2, 1))
	T = find_T_matrix(pts2ptsh(pts), pph)
	H = perspective_transform((dim, dim), angles=angles)
	H = np.matmul(H, T)
	# 将图像I扭曲变换为指定维度dim的图像
	Iroi, pts = project(I, H, pts, dim)
	hsv_mod = np.random.rand(3).astype('float32')
	hsv_mod = (hsv_mod - .5) * .3
	hsv_mod[0] *= 360
	Iroi = hsv_transform(Iroi, hsv_mod)
	Iroi = np.clip(Iroi, 0., 1.)
	pts = np.array(pts)
	if random.random() > .5:
		Iroi, pts = flip_image_and_pts(Iroi, pts)
	tl, br = pts.min(1), pts.max(1)
	llp = Label(0, tl, br)
	return Iroi, llp, pts
