# -*- encoding: utf-8 -*-
'''
----------------------------
File       : 12_densenet.py
Data       : 2021-03-22
Author     : stxws
----------------------------
Description :
	dense net
'''

import os
import time
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss

# 稠密块
def conv_block(num_channels):
	blk = nn.Sequential()
	blk.add(
		nn.BatchNorm(),
		nn.Activation('relu'),
		nn.Conv2D(num_channels, kernel_size=3, padding=1)
	)
	return blk

class DenseBlock(nn.Block):
	def __init__(self, num_convs, num_channels, **kwargs):
		super(DenseBlock, self).__init__(**kwargs)
		self.net = nn.Sequential()
		for _ in range(num_convs):
			self.net.add(conv_block(num_channels))
	
	def forward(self, x):
		for blk in self.net:
			y = blk(x)
			x = nd.concat(x, y, dim=1)
		return x

blk = DenseBlock(2, 10)
blk.initialize()
x = nd.random.uniform(shape=(4, 3, 8, 8))
y = blk(x)
print(y.shape)


# 过渡层
def transition_block(num_channels):
	blk = nn.Sequential()
	blk.add(
		nn.BatchNorm(),
		nn.Activation('relu'),
		nn.Conv2D(num_channels, kernel_size=1),
		nn.AvgPool2D(pool_size=2, strides=2)
	)
	return blk

blk = transition_block(10)
blk.initialize()
y = blk(y)
print(y.shape)

net = nn.Sequential()
net.add(
	nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
	nn.BatchNorm(),
	nn.Activation('relu'),
	nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)


# DenseNet模型
net = nn.Sequential()
net.add(
	nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
	nn.BatchNorm(),
	nn.Activation('relu'),
	nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

num_channels = 64 # num_channels为当前的通道数
growth_rate = 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
	net.add(DenseBlock(num_convs, growth_rate))
	# 上一个稠密块的输出通道数
	num_channels += num_convs * growth_rate
	# 在稠密块之间加入通道数减半的过渡层
	if i != len(num_convs_in_dense_blocks) - 1:
		num_channels //= 2
		net.add(transition_block(num_channels))

net.add(
	nn.BatchNorm(),
	nn.Activation('relu'),
	nn.GlobalAvgPool2D(),
	nn.Dense(10)
)

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
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx)
