# -*- encoding: utf-8 -*-
'''
----------------------------
File       : 11_resnet.py
Data       : 2021-03-22
Author     : stxws
----------------------------
Description :
	残差网络
'''

import os
import time
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss

# 残差块
class Residual(nn.Block):
	'''
	残差模块
	'''
	def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
		super(Residual, self).__init__(**kwargs)

		self.conv_1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
		self.conv_2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
		if use_1x1conv:
			self.conv_3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
		else:
			self.conv_3 = None
		
		self.bn_1 = nn.BatchNorm()
		self.bn_2 = nn.BatchNorm()
	
	def forward(self, x):
		y = self.conv_1(x)
		y = self.bn_1(y)
		y = nd.relu(y)
		
		y = self.conv_2(y)
		y = self.bn_2(y)

		if self.conv_3 != None:
			x = self.conv_3(x)
		
		y = nd.relu(x + y)
		return y

blk = Residual(3)
blk.initialize()
x = nd.random.uniform(shape=(4, 3, 6, 6))
y = blk(x)
print(y.shape)

blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
y = blk(x)
print(y.shape)


# ResNet模型
net = nn.Sequential()
net.add(
	nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
	nn.BatchNorm(),
	nn.Activation('relu'),
	nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

def resnet_block(num_channels, num_residuals, first_block=False):
	blk = nn.Sequential()
	for i in range(num_residuals):
		if i == 0 and first_block == False:
			blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
		else:
			blk.add(Residual(num_channels))
	return blk

net.add(
	resnet_block(64, 2, first_block=True),
	resnet_block(128, 2),
	resnet_block(256, 2),
	resnet_block(512, 2)
)
net.add(
	nn.GlobalAvgPool2D(),
	nn.Dense(10)
)

x = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
	x = layer(x)
	print(("%-12s output shape:" % (layer.name)), x.shape)


# 训练模型
def try_gpu(gpu_id):
	try:
		ctx = mx.gpu(gpu_id)
		_ = nd.zeros((1,), ctx=ctx)
	except mx.base.MXNetError:
		ctx = mx.cpu()
	return ctx

def load_data_fashion_mnist(batch_size, resize=None, 
		root=os.path.join("~", ".mxnet", "dataset", "fashion-mnist")):
	root = os.path.expanduser(root)
	transformer = []
	if resize:
		transformer += [g_data.vision.transforms.Resize(resize)]
	transformer += [g_data.vision.transforms.ToTensor()]
	transformer = g_data.vision.transforms.Compose(transformer)
	mnist_train = g_data.vision.FashionMNIST(root=root, train=True)
	mnist_test = g_data.vision.FashionMNIST(root=root, train=False)
	# num_workers = 0 if sys.platform.startswith('win32') else 4
	train_iter = g_data.DataLoader(
		mnist_train.transform_first(transformer), batch_size=batch_size, shuffle=True)
	test_iter = g_data.DataLoader(
		mnist_test.transform_first(transformer), batch_size=batch_size, shuffle=False)
	return train_iter, test_iter

def evaluate_accuracy(data_iter, net, ctx):
	acc_sum, n = nd.array([0], ctx=ctx), 0
	for x, y in data_iter:
		x, y = x.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
		acc_sum += (net(x).argmax(axis=1) == y).sum()
		n += y.size
		return acc_sum.asscalar() / n

def train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx=mx.cpu()):
	print("training on", ctx)
	loss = g_loss.SoftmaxCrossEntropyLoss()
	for epoch in range(num_epochs):
		train_l_sum = 0.0
		train_acc_sum = 0.0
		n = 0
		start = time.time()
		for x, y in train_iter:
			x = x.as_in_context(ctx)
			y = y.as_in_context(ctx)
			with autograd.record():
				y_hat = net(x)
				l = loss(y_hat, y)
			l.backward()
			trainer.step(batch_size)
			train_l_sum += l.sum().asscalar()

			y = y.astype('float32')
			train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
			n += y.size
		test_acc = evaluate_accuracy(test_iter, net, ctx)
		print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
			% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc, 
			time.time() - start))

lr = 0.05
num_epochs = 5
batch_size = 256
ctx = try_gpu(5)
net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx)
