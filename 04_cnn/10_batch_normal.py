# coding=utf-8
import os, sys
import time
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss


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


# 从零开始实现
def batch_norm(x, gamma, beta, moving_mean, moving_var, momentum=0.9, eps=1e-5):
	if not autograd.is_training():
		x_hat = (x - moving_mean) / nd.sqrt(moving_var + eps)
	else:
		assert len(x.shape) in (2, 4)
		if len(x.shape) == 2:
			mean = x.mean(axis=0)
			var = ((x - mean) ** 2).mean(axis=0)
		else:
			mean = x.mean(axis=(0, 2, 3), keepdims=True)
			var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
		x_hat = (x - mean) / nd.sqrt(var + eps)
		moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
		moving_var = momentum * moving_var + (1.0 - momentum) * var
	y = gamma * x_hat + beta
	return y, moving_mean, moving_var

class BatchNorm(nn.Block):
	def __init__(self, num_features, num_dims, **kwargs):
		super(BatchNorm, self).__init__(**kwargs)
		if num_dims == 2:
			shape = (1, num_features)
		else:
			shape = (1, num_features, 1, 1)
		self.gamma = self.params.get("gamma", shape=shape, init=init.One())
		self.beta = self.params.get("beta", shape=shape, init=init.Zero())
		self.moving_mean = nd.zeros(shape)
		self.moving_var = nd.zeros(shape)
	
	def forward(self, x):
		if self.moving_mean.context != x.context:
			self.moving_mean = self.moving_mean.copyto(x.context)
			self.moving_var = self.moving_var.copyto(x.context)
		y, self.moving_mean, self.moving_var = batch_norm(
			x, self.gamma.data(), self.beta.data(), self.moving_mean, self.moving_var)
		return y

# net = nn.Sequential()
# net.add(
# 	nn.Conv2D(6, kernel_size=5),
# 	BatchNorm(6, num_dims=4),
# 	nn.Activation('sigmoid'),
# 	nn.MaxPool2D(pool_size=2, strides=2),
# 	nn.Conv2D(16, kernel_size=5),
# 	BatchNorm(16, num_dims=4),
# 	nn.Activation('sigmoid'),
# 	nn.MaxPool2D(pool_size=2, strides=2),
# 	nn.Dense(120),
# 	BatchNorm(120, num_dims=2),
# 	nn.Activation('sigmoid'),
# 	nn.Dense(84),
# 	BatchNorm(84, num_dims=2),
# 	nn.Activation('sigmoid'),
# 	nn.Dense(10)
# )

# lr = 1.0
# num_epochs = 5
# batch_size = 256
# ctx = try_gpu(7)
# net.initialize(init=init.Xavier(), ctx=ctx)
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
# train_iter, test_iter = load_data_fashion_mnist(batch_size)
# train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx)


# 简洁实现
net = nn.Sequential()
net.add(
	nn.Conv2D(6, kernel_size=5),
	nn.BatchNorm(),
	nn.Activation('sigmoid'),
	nn.MaxPool2D(pool_size=2, strides=2),
	nn.Conv2D(16, kernel_size=5),
	nn.BatchNorm(),
	nn.Activation('sigmoid'),
	nn.MaxPool2D(pool_size=2, strides=2),
	nn.Dense(120),
	nn.BatchNorm(),
	nn.Activation('sigmoid'),
	nn.Dense(84),
	nn.BatchNorm(),
	nn.Activation('sigmoid'),
	nn.Dense(10)
)

lr = 1.0
num_epochs = 5
batch_size = 256
ctx = try_gpu(7)
net.initialize(init=init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train(net, train_iter, test_iter, batch_size, trainer, num_epochs, ctx)
