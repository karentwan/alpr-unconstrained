import sys
import cv2
import traceback
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes
import time


def adjust_pts(pts, lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':
	try:
		# input_dir  = sys.argv[1]
		# output_dir = input_dir
		input_dir = '/home/tang/pyprojects/alpr/result/100_version2'
		output_dir = '/home/tang/pyprojects/alpr/result/output_dir'
		lp_threshold = .5
		wpod_net_path = './models/eccv-model-scracth.h5'
		model_path = './models/eccv-model-scracth.json'
		print('wpod_net_path:{}'.format(wpod_net_path))
		wpod_net = load_model(model_path, wpod_net_path)
		imgs_paths = glob('%s/*.jpg' % input_dir)
		print('Searching for license plates using WPOD-NET')
		total_time = 0
		for i, img_path in enumerate(imgs_paths):
			print('\t Processing %s' % img_path)
			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)
			# Ivehicle.shape = [h, w, c], Ivehicle.shape[:2] = [h, w]
			# 下面对应论文公式(1), 图像越小, 需要放大更大的倍数
			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side = int(ratio*288.)
			bound_dim = min(side + (side % (2**4)), 608)
			print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
			print('Ivehicle.shape:{}'.format(Ivehicle.shape))
			start = time.time()
			Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold)
			duration = time.time() - start
			total_time += duration
			# print('duration:{}'.format(duration, .4f))
			print('duration:%.4f' % duration)
			if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
				s = Shape(Llp[0].pts)
				cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])
		print('mean time :%.4f'% (total_time / 100))
	except:
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)


