import tensorflow as tf


def logloss(Ptrue, Pred, szs, eps=10e-10):
	b, h, w, ch = szs
	Pred = tf.clip_by_value(Pred, eps, 1.)
	Pred = -tf.log(Pred)
	Pred = Pred * Ptrue
	Pred = tf.reshape(Pred, (b, h * w * ch))
	Pred = tf.reduce_sum(Pred, 1)
	return Pred


def l1(true, pred, szs):
	b, h, w, ch = szs
	res = tf.reshape(true-pred, (b, h*w*ch))
	res = tf.abs(res)
	res = tf.reduce_sum(res, 1)
	return res


# 对应论文里面的公式(2)(3)(4)(5)(6)
def loss(Ytrue, Ypred):
	# Yture是一个shape=[M, N, 9]的feature map，
	# 第一个feature map的值有车牌的位置值为1，没有车牌的位置值为0，剩余的8个feature map分别是ltx, rtx, rbx, lbx, lty, rty, rby, lby
	# Ypred = [M, N, 8]
	b = tf.shape(Ytrue)[0]
	h = tf.shape(Ytrue)[1]
	w = tf.shape(Ytrue)[2]
	# ...省略前面所有的冒号，切片中使用
	# 即Ytrue[..., 0] = YTrue[:, :, :, 0]
	# 网络的输出一共8个通道，前两个通道作为分类，剩余的6个通道来做仿射变换
	obj_probs_true = Ytrue[..., 0]
	obj_probs_pred = Ypred[..., 0]
	non_obj_probs_true = 1. - Ytrue[..., 0]
	non_obj_probs_pred = Ypred[..., 1]
	# 6个通道的值作为仿射变换的值
	affine_pred = Ypred[..., 2:]
	pts_true = Ytrue[..., 1:]  # 获得车牌所在位置的坐标，当有车牌的时候值为1，没有车牌的时候值为0
	affinex = tf.stack([tf.maximum(affine_pred[..., 0], 0.), affine_pred[..., 1], affine_pred[..., 2]], 3)
	affiney = tf.stack([affine_pred[..., 3], tf.maximum(affine_pred[..., 4], 0.), affine_pred[..., 5]], 3)
	# 构建矩阵q，这是一个规范的矩形，后面将其进行仿射变换
	v = 0.5
	# 这里的12个维度，分别是4个点的坐标加上4个1 = 2 * 4 + 4
	base = tf.stack([[[[-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.]]]])  # base = [1, 1, 1, 12]
	base = tf.tile(base, tf.stack([b, h, w, 1]))  # shape = [b, h, w, 12]
	pts = tf.zeros((b, h, w, 0))
	for i in range(0, 12, 3):  # 0-12遍历，每隔3一次，即i = 0, 3, 6, 9
		row = base[..., i:(i+3)]  # row = [b, h, w, 3], 这里的3对应max(v3, 0), v4, v5
		ptsx = tf.reduce_sum(affinex*row, 3)  # 矩阵乘法，对应位置相乘再相加，shape = [b, h, w, 1]
		ptsy = tf.reduce_sum(affiney*row, 3)  #
		pts_xy = tf.stack([ptsx, ptsy], 3)
		pts = (tf.concat([pts, pts_xy], 3))
	flags = tf.reshape(obj_probs_true, (b, h, w, 1))
	res = 1. * l1(pts_true * flags, pts * flags, (b, h, w, 4*2))
	res += 1. * logloss(obj_probs_true, obj_probs_pred, (b, h, w, 1))
	res += 1. * logloss(non_obj_probs_true, non_obj_probs_pred, (b, h, w, 1))
	return res
