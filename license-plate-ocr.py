import sys
import traceback
import darknet.python.darknet as dn
from os.path import splitext, basename
from glob import glob
from darknet.python.darknet import detect
from src.label import dknet_label_conversion
from src.utils import nms
import time


if __name__ == '__main__':
	try:
		input_dir = '/home/tang/pyprojects/alpr/result/output_dir'
		output_dir = input_dir
		ocr_threshold = .4
		ocr_weights = 'data/ocr/ocr-net.weights'.encode('utf-8')
		ocr_netcfg = 'data/ocr/ocr-net.cfg'.encode('utf-8')
		ocr_dataset = 'data/ocr/ocr-net.data'.encode('utf-8')
		ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
		ocr_meta = dn.load_meta(ocr_dataset)
		imgs_paths = sorted(glob('%s/*.png' % output_dir))
		print('Performing OCR...')
		total_time = 0
		for i,img_path in enumerate(imgs_paths):
			img_path = img_path.encode('utf-8')
			print('\tScanning %s' % img_path)
			bname = basename(splitext(img_path)[0])
			start = time.time()
			R, (width, height) = detect(ocr_net, ocr_meta, img_path, thresh=ocr_threshold, nms=None)
			duration = time.time() - start
			total_time += duration
			if len(R):
				L = dknet_label_conversion(R, width, height)
				L = nms(L, .45)
				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])
				with open('%s/%s_str.txt' % (output_dir, bname), 'w') as f:
					f.write(lp_str + '\n')
				print('\t\tLP: %s' % lp_str)
			else:
				print('No characters found')
		print('mean time:%.4f'%(total_time / 100))
	except:
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)
