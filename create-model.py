import sys
import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Add, Activation, Concatenate, Input
from keras.models import Model
from keras.applications.mobilenet import MobileNet
from src.keras_utils import save_model


def res_block(x, sz, filter_sz=3, in_conv_size=1):
	xi = x
	for i in range(in_conv_size):
		xi = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
		xi = BatchNormalization()(xi)
		xi = Activation('relu')(xi)
	xi = Conv2D(sz, filter_sz, activation='linear', padding='same')(xi)
	xi = BatchNormalization()(xi)
	xi = Add()([xi, x])
	xi = Activation('relu')(xi)
	return xi


def conv_batch(_input, fsz, csz, activation='relu', padding='same', strides=(1, 1)):
	'''
	:param _input:
	:param fsz: 输出卷积通道的数量
	:param csz: 卷积核的大小
	:param activation: 激活函数
	:param padding: 填充方式
	:param strides: 补偿
	:return:
	'''
	output = Conv2D(fsz, csz, activation='linear', padding=padding, strides=strides)(_input)
	output = BatchNormalization()(output)
	output = Activation(activation)(output)
	return output


def end_block(x):
	xprobs = Conv2D(2, 3, activation='softmax', padding='same')(x)
	xbbox = Conv2D(6, 3, activation='linear', padding='same')(x)
	return Concatenate(3)([xprobs, xbbox])


# 调用这个函数
def create_model_eccv():
	# shape = [32, 208, 208, 3], 这个类似于placeholder
	input_layer = Input(shape=(None, None, 3), name='input')
	x = conv_batch(input_layer, 16, 3)  # conv->BN->Relu
	# conv_batch(x, 16, 3), out_channel = 16, kernel_size = 3
	x = conv_batch(x, 16, 3)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = conv_batch(x, 32, 3)
	x = res_block(x, 32)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = conv_batch(x, 64, 3)
	x = res_block(x, 64)
	x = res_block(x, 64)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = conv_batch(x, 64, 3)
	x = res_block(x, 64)
	x = res_block(x, 64)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = conv_batch(x, 128, 3)
	x = res_block(x, 128)
	x = res_block(x, 128)
	x = res_block(x, 128)
	x = res_block(x, 128)
	x = end_block(x)
	# 这一步必不可少
	return Model(inputs=input_layer, outputs=x)


# Model not converging...
def create_model_mobnet():
	input_layer = Input(shape=(None, None, 3), name='input')
	x = input_layer
	mbnet = MobileNet(input_shape=(224, 224, 3), include_top=True)
	backbone = keras.models.clone_model(mbnet)
	for i, bblayer in enumerate(backbone.layers[1:74]):
		layer = bblayer.__class__.from_config(bblayer.get_config())
		layer.name = 'backbone_' + layer.name
		x = layer(x)
	x = end_block(x)
	model = Model(inputs=input_layer, outputs=x)
	backbone_layers = {'backbone_' + layer.name: layer for layer in backbone.layers}
	for layer in model.layers:
		if layer.name in backbone_layers:
			print('setting ' + layer.name)
			layer.set_weights(backbone_layers[layer.name].get_weights())
	return model


# python create-model.py eccv models/eccv-model-scracth
if __name__ == '__main__':
	modules = [func.replace('create_model_', '') for func in dir(sys.modules[__name__]) if 'create_model_' in func]
	# assert sys.argv[1] in modules, \
	# 	'Model name must be on of the following: %s' % ', '.join(modules)
	# sys.argv[1]=eccv, sys.argv[2]=models/eccv-model-scratch
	modelf = getattr(sys.modules[__name__], 'create_model_' + 'eccv')
	print('Creating model %s' % 'eccv')
	model = modelf()
	print('Finished')
	print('Saving at %s' % 'models/eccv-model-scratch')
	# 以json的形式保存模型结构到文件以及保存模型本身到磁盘
	save_model(model, 'models/eccv-model-scratch')
