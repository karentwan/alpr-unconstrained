
import sys
import numpy as np
import cv2
import argparse
import keras
import tensorflow as tf
from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder, show
from src.sampler import augment_sample, labels2output_map
from src.data_generator import DataGenerator

from pdb import set_trace as pause


def load_network(modelpath, lp_path, input_dim):

	model = load_model(modelpath, lp_path)
	input_shape = (input_dim, input_dim, 3)

	# Fixed input size for training
	inputs  = keras.layers.Input(shape=(input_dim,input_dim,3))
	outputs = model(inputs)
	# output_shape = [h, w, c], 也就是输出的feature map的shape
	# 注意，这里的输出模型是8个通道
	output_shape = tuple([s.value for s in outputs.shape[1:]])
	output_dim   = output_shape[1]
	model_stride = input_dim / output_dim

	assert input_dim % output_dim == 0, \
		'The output resolution must be divisible by the input resolution'

	assert model_stride == 2**4, \
		'Make sure your model generates a feature map with resolution ' \
		'16x smaller than the input'

	return model, model_stride, input_shape, output_shape


def process_data_item(data_item, dim, model_stride):
	XX, llp, pts = augment_sample(data_item[0],data_item[1].pts,dim)
	YY = labels2output_map(llp, pts, dim, model_stride)
	return XX, YY


def write_log(callback, names, logs, batch_no):
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()


# 训练WPOD-NET网络，该网络用来检测车牌并纠正，不涉及到车牌识别
if __name__ == '__main__':
	# python train-detector.py --model models/eccv-model-scracth --name my-trained-model
	# --train-dir samples/train-detector --output-dir models/my-trained-model/
	# -op Adam -lr .001 -its 300000 -bs 64
	parser = argparse.ArgumentParser()
	parser.add_argument('-m' 		,'--model'			,type=str   , default='models/eccv-model-scracth.json', help='Path to previous model')
	parser.add_argument('-lp' 		,'--lp_path'			,type=str   , default='models/eccv-model-scracth.h5', help='Path to previous model')
	parser.add_argument('-n' 		,'--name'			,type=str   , default='my-trained-model', help='Model name')
	parser.add_argument('-tr'		,'--train-dir'		,type=str   , default='samples/train-detector', help='Input data directory for training')
	parser.add_argument('-its'		,'--iterations'		,type=int   , default=400000	, help='Number of mini-batch iterations (default = 300.000)')
	parser.add_argument('-bs'		,'--batch-size'		,type=int   , default=32		, help='Mini-batch size (default = 32)')
	parser.add_argument('-od'		,'--output-dir'		,type=str   , default='models/my-trained-model/', help='Output directory (default = ./)')
	parser.add_argument('-op'		,'--optimizer'		,type=str   , default='Adam'	, help='Optmizer (default = Adam)')
	parser.add_argument('-lr'		,'--learning-rate'	,type=float , default=.001		, help='Optmizer (default = 0.01)')
	args = parser.parse_args()

	netname = basename(args.name)
	train_dir = args.train_dir
	outdir = args.output_dir

	iterations = args.iterations
	batch_size = args.batch_size
	dim = 208

	if not isdir(outdir):
		makedirs(outdir)

	model, model_stride, xshape, yshape = load_network(args.model, args.lp_path, dim)
	# 获取要使用的优化器，这里使用的是Adam优化器
	opt = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
	model.compile(loss=loss, optimizer=opt)

	print('Checking input directory...')
	# 获取文件夹下面的所有图片文件名列表
	Files = image_files_from_folder(train_dir)
	# Data由图片和Label组成，形式如下：[[img, label], [img, label]]
	Data = []
	for file in Files:
		# 获取跟图片同名的txt文件，然后读取其中的数据
		labfile = splitext(file)[0] + '.txt'
		if isfile(labfile):
			L = readShapes(labfile)
			I = cv2.imread(file)
			Data.append([I, L[0]])

	print('%d images with labels found' % len(Data))

	dg = DataGenerator(data=Data,
					   process_data_item_func=lambda x: process_data_item(x,dim,model_stride),
					   xshape=xshape,
					   yshape=(yshape[0], yshape[1], yshape[2]+1),
					   nthreads=2,
					   pool_size=1000,
					   min_nsamples=100)
	dg.start()
	# np.empty()函数会返回一个指定维度的随机矩阵
	Xtrain = np.empty((batch_size, dim, dim, 3), dtype='single')
	Ytrain = np.empty((batch_size, int(dim//model_stride), int(dim//model_stride), 2*4+1))

	model_path_backup = '%s/%s_backup' % (outdir, netname)
	model_path_final = '%s/%s_final' % (outdir, netname)
	# record loss function
	writer = tf.summary.FileWriter("/home/tang/experimental/alpr", tf.get_default_graph())
	sess = tf.Session()
	for it in range(iterations):

		print('Iter. %d (of %d)' % (it+1, iterations))

		Xtrain, Ytrain = dg.get_batch(batch_size)
		train_loss = model.train_on_batch(Xtrain, Ytrain)

		print('\tLoss: %f' % train_loss)

		####################### add
		summary = tf.Summary(value=[
			tf.Summary.Value(tag='loss', simple_value=train_loss)
		])
		writer.add_summary(summary)
		# sess.run(summary)
		######################### end

		# Save model every 1000 iterations
		if (it+1) % 1000 == 0:
			print('Saving model (%s)' % model_path_backup)
			save_model(model, model_path_backup)

	print('Stopping data generator')
	dg.stop()

	print('Saving model (%s)' % model_path_final)
	save_model(model, model_path_final)
