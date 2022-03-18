# coding=utf-8
import os, sys
import time
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss

# Inception块
class Inception(nn.Block):
	def __init__(self, c1, c2, c3, c4, **kwargs):
		super(Inception, self).__init__(**kwargs)
		# 线路1，单1x1卷积层
		self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
		# 线路2，1x1卷积层后接3x3卷积层
		self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
		self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')
		# 线路3，1x1卷积层后接5x5卷积层
		self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
		self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')
		# 线路4，3x3最大池化层后接1x1卷积层
		self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
		self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')
	
	def forward(self, x):
		p1 = self.p1_1(x)
		p2 = self.p2_2(self.p2_1(x))
		p3 = self.p3_2(self.p3_1(x))
		p4 = self.p4_2(self.p4_1(x))
		return nd.concat(p1, p2, p3, p4, dim=1)

# GoogLeNet模型
b1 = nn.Sequential()
b1.add(
	nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
	nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

b2 = nn.Sequential()
b2.add(
	nn.Conv2D(64, kernel_size=1, activation='relu'),
	nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
	nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

b3 = nn.Sequential()
b3.add(
	Inception(64, (96, 128), (16, 32), 32),
	Inception(128, (128, 192), (32, 96), 64),
	nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

b4 = nn.Sequential()
b4.add(
	Inception(192, (96, 208), (16, 48), 64),
	Inception(160, (112, 224), (24, 64), 64),
	Inception(128, (128, 256), (24, 64), 64),
	Inception(112, (144, 224), (32, 64), 64),
	Inception(256, (160, 320), (32, 128), 128),
	nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)

b5 = nn.Sequential()
b5.add(
	Inception(256, (160, 320), (32, 128), 128),
	Inception(384, (192, 384), (48, 128), 128),
	nn.GlobalAvgPool2D()
)

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))

x = nd.random.uniform(shape=(1, 1, 96, 96))
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

lr = 0.1
num_epochs = 5
batch_size = 128
ctx = try_gpu(5)
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx=ctx)
