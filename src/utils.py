import numpy as np
import cv2
import sys
from glob import glob


def im2single(I):
	assert(I.dtype == 'uint8')
	return I.astype('float32')/255.


def getWH(shape):
	return np.array(shape[1::-1]).astype(float)


# Intersection over Union，计算交并比，如果是算预测出来的框和ground truth之间的框，那么就是衡量预测准确性的指标
# tl1和br1分别是第一个框的左上和右下坐标，tl2和br2是第二个框的左上和右下
def IOU(tl1, br1, tl2, br2):
	wh1, wh2 = br1-tl1, br2-tl2
	assert((wh1 >= .0).all() and (wh2 >= .0).all())
	# a = [21, 35], b = [24, 36], np.minimum(a, b) = [21, 35]
	# a = [18, 35], b = [24, 36], np.minimum(a, b) = [18, 35]
	intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)  # 求两个矩形的交集
	# np.prod连乘操作
	intersection_area = np.prod(intersection_wh)
	area1, area2 = (np.prod(wh1), np.prod(wh2))
	union_area = area1 + area2 - intersection_area  # 求两个矩形的并集
	return intersection_area/union_area


def IOU_labels(l1, l2):
	return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())


def IOU_centre_and_dims(cc1, wh1, cc2, wh2):
	return IOU(cc1-wh1/2., cc1+wh1/2., cc2-wh2/2., cc2+wh2/2.)


# 非极大值抑制
def nms(Labels, iou_threshold=.5):
	# iou_threshold = 0.1
	SelectedLabels = []
	# 先根据概率(也就是使用第一个feature map的值)来排序
	Labels.sort(key=lambda l: l.prob(), reverse=True)
	# 思想：首先取第一个值放入SelectedLabels数组中, 然后将跟这个数组里面所有元素都不相交的
	# 的框放入数组中
	for label in Labels:
		non_overlap = True
		for sel_label in SelectedLabels:
			# IOU大于0.1的都不要
			if IOU_labels(label, sel_label) > iou_threshold:
				non_overlap = False
				break
		if non_overlap:
			SelectedLabels.append(label)
	return SelectedLabels


def image_files_from_folder(folder, upper=True):
	extensions = ['jpg', 'jpeg', 'png']
	img_files = []
	for ext in extensions:
		img_files += glob('%s/*.%s' % (folder, ext))
		if upper:
			img_files += glob('%s/*.%s' % (folder, ext.upper()))
	return img_files


def is_inside(ltest, lref):
	return (ltest.tl() >= lref.tl()).all() and (ltest.br() <= lref.br()).all()


def crop_region(I, label, bg=0.5):
	wh = np.array(I.shape[1::-1])
	ch = I.shape[2] if len(I.shape) == 3 else 1
	tl = np.floor(label.tl()*wh).astype(int)
	br = np.ceil(label.br()*wh).astype(int)
	outwh = br-tl
	if np.prod(outwh) == 0.:
		return None
	outsize = (outwh[1], outwh[0], ch) if ch > 1 else (outwh[1], outwh[0])
	if (np.array(outsize) < 0).any():
		# pause()
		input()
	Iout = np.zeros(outsize, dtype=I.dtype) + bg
	offset = np.minimum(tl, 0)*(-1)
	tl = np.maximum(tl, 0)
	br = np.minimum(br, wh)
	wh = br - tl
	Iout[offset[1]:(offset[1] + wh[1]), offset[0]:(offset[0] + wh[0])] = I[tl[1]:br[1], tl[0]:br[0]]
	return Iout


def hsv_transform(I, hsv_modifier):
	I = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
	I = I + hsv_modifier
	return cv2.cvtColor(I, cv2.COLOR_HSV2BGR)


def IOU(tl1, br1, tl2, br2):
	wh1, wh2 = br1-tl1, br2-tl2
	assert((wh1 >= .0).all() and (wh2 >= .0).all())
	intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0.)
	intersection_area = np.prod(intersection_wh)
	area1, area2 = (np.prod(wh1), np.prod(wh2))
	union_area = area1 + area2 - intersection_area
	return intersection_area/union_area

# 传入中心点和宽高, 并以此计算IOU
def IOU_centre_and_dims(cc1, wh1, cc2, wh2):
	return IOU(cc1-wh1/2., cc1+wh1/2., cc2-wh2/2., cc2+wh2/2.)


def show(I, wname='Display'):
	cv2.imshow(wname, I)
	cv2.moveWindow(wname, 0, 0)
	key = cv2.waitKey(0) & 0xEFFFFF
	cv2.destroyWindow(wname)
	if key == 27:
		sys.exit()
	else:
		return key
