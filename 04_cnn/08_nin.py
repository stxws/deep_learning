#coding=utf-8
import os, sys
import time
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss

# NiN块
def nin_block(num_channels, kernel_size, strides, padding):
	blk = nn.Sequential()
	blk.add(
		nn.Conv2D(num_channels, kernel_size,
			strides=strides, padding=padding, activation='relu'),
		nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
		nn.Conv2D(num_channels, kernel_size=1, activation='relu')
	)
	return blk

# NiN模型
net = nn.Sequential()
net.add(
	nin_block(96, kernel_size=11, strides=4, padding=0),
	nn.MaxPool2D(pool_size=3, strides=2),
	nin_block(256, kernel_size=5, strides=1, padding=2),
	nn.MaxPool2D(pool_size=3, strides=2),
	nin_block(384, kernel_size=3, strides=1, padding=1),
	nn.MaxPool2D(pool_size=3, strides=2),
	nn.Dropout(0.5),
	# 标签类别数是10
	nin_block(10, kernel_size=3, strides=1, padding=1),
	# 全局平均池化层将窗口形状自动设置成输入的高和宽
	nn.GlobalAvgPool2D(),
	# 将四维的输出转成二维的输出，其形状为(批量大小, 10)
	nn.Flatten()
)

x = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
	x = layer(x)
	print(layer.name, 'output shape:\t', x.shape)

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
	num_workers = 0 if sys.platform.startswith('win32') else 4
	train_iter = g_data.DataLoader(
		mnist_train.transform_first(transformer), batch_size=batch_size,
		shuffle=True, num_workers=num_workers)
	test_iter = g_data.DataLoader(
		mnist_test.transform_first(transformer), batch_size=batch_size,
		shuffle=False, num_workers=num_workers)
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

lr = 0.1
num_epochs = 5
batch_size = 128
ctx = try_gpu(7)
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx=ctx)
