import numpy as np
from threading import Semaphore, Thread
from time import sleep
from random import choice, randint
import cv2
from pdb import set_trace as pause


class DataGenerator(object):

	def __init__(self, data, process_data_item_func, xshape, yshape,
				data_item_selector =choice,
				nthreads =2,
				pool_size =1000,
				min_nsamples =1,
				dtype  ='single'):

		assert pool_size >= min_nsamples, \
			'Min. samples must be equal or less than pool_size'
		assert min_nsamples > 0 and pool_size > 0, \
			'Min. samples and pool size must be positive non-zero numbers'

		self._data = data
		self._process_data_item = process_data_item_func
		self._data_item_selector = data_item_selector
		self._xshape = xshape
		self._yshape = yshape  # yshape = xshape / stride
		self._nthreads = nthreads  # 线程的数量
		self._pool_size = pool_size
		self._min_nsamples = min_nsamples
		self._dtype = dtype
		
		self._count = 0
		self._stop = False
		self._threads = []
		self._sem = Semaphore()

		self._X, self._Y = self._get_buffers(self._pool_size)

	def _get_buffers(self, N):
		# N就是batch_size的大小，默认N = 1000, self._xshape = (208, 208, 3), self._dtype=single
		X = np.empty((N,) + self._xshape, dtype=self._dtype)
		# self._yshape = (13, 13, 8)
		Y = np.empty((N,) + self._yshape, dtype=self._dtype)
		# x = (1000, 208, 208, 3)
		# y = (1000, 13, 13, 8)
		# (1000, ) + (208, 208, 3) = (1000, 208, 208, 3) 元组相加小知识
		return X, Y

	def _compute_sample(self):
		# 从self._data数组中随机选择一个数据
		d = self._data_item_selector(self._data)
		return self._process_data_item((cv2.imread(d[0]), d[1]))

	def _insert_data(self, x, y):
		self._sem.acquire()

		if self._count < self._pool_size:
			self._X[self._count] = x
			self._Y[self._count] = y
			self._count += 1
		else:
			idx = randint(0, self._pool_size-1)
			self._X[idx] = x
			self._Y[idx] = y

		self._sem.release()

	def _run(self):
		while True:
			x, y = self._compute_sample()  # 读取数据
			self._insert_data(x, y)        # 往缓存里面插入数据
			if self._stop:
				break

	def stop(self):
		self._stop = True
		for thread in self._threads:
			thread.join()

	def start(self):
		self._stop = False
		self._threads = [Thread(target=self._run) for n in range(self._nthreads)]
		for thread in self._threads:
			thread.setDaemon(True)
			thread.start()  # 开启多线程

	def get_batch(self, N):
		# Wait until the buffer was filled with the minimum
		# number of samples
		while self._count < self._min_nsamples:
			sleep(.1)

		X, Y = self._get_buffers(N)
		self._sem.acquire()
		# 随机取batch个数据，索引是随机产生的
		for i in range(N):
			idx = randint(0, self._count-1)
			X[i] = self._X[idx]
			Y[i] = self._Y[idx]
		self._sem.release()
		return X, Y


