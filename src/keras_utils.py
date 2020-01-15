import numpy as np
import cv2
import time
from os.path import splitext
from src.label import Label
from src.utils import getWH, nms
from src.projection_utils import getRectPts, find_T_matrix


class DLabel (Label):

	def __init__(self, cl, pts, prob):
		self.pts = pts
		tl = np.amin(pts, 1)
		br = np.amax(pts, 1)
		Label.__init__(self, cl, tl, br, prob)


def save_model(model, path, verbose=0):
	path = splitext(path)[0]
	model_json = model.to_json()
	with open('%s.json' % path, 'w') as json_file:
		json_file.write(model_json)
	model.save_weights('%s.h5' % path)
	if verbose: print('Saved to %s' % path)


def load_model(path, weight, custom_objects={}, verbose=0):
	from keras.models import model_from_json
	with open(path, 'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights(weight)
	if verbose: print('Loaded from %s' % path)
	return model


def reconstruct(Iorig, I, Y, out_size, threshold=.9):
	# Iorig原始图像
	# resize后的Iorig
	# Y 是维度为[M, N, 8]的feature map, 没有b, 三个维度
	# threshold = 0.5, 预测车牌所在的区域
	net_stride = 2 ** 4
	side = ((208. + 40.)/2.) / net_stride  # 7.75
	Probs = Y[..., 0]
	Affines = Y[..., 2:]
	rx, ry = Y.shape[:2]
	ywh = Y.shape[1::-1]
	iwh = np.array(I.shape[1::-1], dtype=float).reshape((2, 1))
	xx, yy = np.where(Probs > threshold)  # 获取概率大于阈值的地方，也就是预测的车牌区域, np.where返回的是大于阈值的坐标
	WH = getWH(I.shape)
	MN = WH / net_stride
	vxx = vyy = 0.5  # alpha
	base = lambda vx, vy: np.matrix([[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
	# base = [3, 4]
	labels = []
	for i in range(len(xx)):
		y, x = xx[i], yy[i]
		affine = Affines[y, x]
		prob = Probs[y, x]
		mn = np.array([float(x) + .5, float(y) + .5])
		A = np.reshape(affine, (2, 3))  # 将A变成一个维度(2, 3)的矩阵
		A[0, 0] = max(A[0, 0], 0.)  # max(v3, 0)
		A[1, 1] = max(A[1, 1], 0.)  # max(v6, 0)
		# 因为base是np.matrix, 因此这里的A * base就是矩阵相乘
		pts = np.array(A * base(vxx, vyy))  # *alpha
		pts_MN_center_mn = pts * side
		# 标签里面减掉了mn的值，所以这里要加上
		pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
		pts_prop = pts_MN / MN.reshape((2, 1))
		labels.append(DLabel(0, pts_prop, prob))
	final_labels = nms(labels, .1)  # labels里面有很多个框，因为预测出来了很多个框，这里使用非极大值抑制去掉多余的框
	TLps = []
	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for i, label in enumerate(final_labels):
			# t_ptsh = [0, 0, 240, 80]
			t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
			# getWH(Iorig.shape).reshape((2, 1))就是获取图像的width和height, 并转置为(2, 1)
			# label.pts = (2, 4)
			# ptsh = (3, 4)
			ptsh = np.concatenate((label.pts * getWH(Iorig.shape).reshape((2, 1)), np.ones((1, 4))))
			H = find_T_matrix(ptsh, t_ptsh)
			# 下面是对Iorig里面车牌区域进行透视变换，最后的输出只有车牌区域
			Ilp = cv2.warpPerspective(Iorig, H, out_size, borderValue=.0)
			TLps.append(Ilp)
	return final_labels, TLps
	

def detect_lp(model, I, max_dim, net_step, out_size, threshold):
	'''
	:param model: wpod-net
	:param I: 输入图像
	:param max_dim:
	:param net_step: 2 ** 4
	:param out_size:  240 * 8, 车牌的大小
	:param threshold: 0.5
	:return:
	'''
	min_dim_img = min(I.shape[:2])
	factor = float(max_dim) / min_dim_img
	w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
	w += (w % net_step != 0) * (net_step - w % net_step)
	h += (h % net_step != 0) * (net_step - h % net_step)
	Iresized = cv2.resize(I, (w, h))
	T = Iresized.copy()
	T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))  # 也可以使用np.expamd_dim()扩展一个维度
	start = time.time()
	Yr = model.predict(T)
	Yr = np.squeeze(Yr)  # 也可以使用Yr[0]
	elapsed = time.time() - start
	L, TLps = reconstruct(I, Iresized, Yr, out_size, threshold)
	return L, TLps, elapsed
