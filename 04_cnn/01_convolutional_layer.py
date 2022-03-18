#!/usr/bin/python3

from mxnet import autograd, nd, gluon
from mxnet.gluon import nn

# 二维互相关运算
def corr2d(x, k):
	h, w = k.shape
	y = nd.zeros((x.shape[0] - h + 1, x.shape[1] - w + 1))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			y[i, j] = (x[i: i + h, j: j + w] * k).sum()
	return y

x = nd.array([[0,1,2], [3,4,5], [6,7,8]])
k = nd.array([[0,1], [2,3]])
print(corr2d(x, k))

# 二维卷积层
class Conv2D(nn.Block):
	def __init__(self, kernel_size, **kwargs):
		super(Conv2D, self).__init__(**kwargs)
		self.weight = self.params.get("weight", shape=kernel_size)
		self.bias = self.params.get("bias", shape=(1,))

	def forward(self, x):
		return corr2d(x, self.weight.data()) + self.bias.data()

# 图像中物体边缘检测
x = nd.ones((6, 8))
x[:, 2:6] = 0
print(x)
k = nd.array([[1, -1]])
y = corr2d(x, k)
print(y)

# 通过数据学习核数组
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

x = x.reshape((1, 1, 6, 8))
y = y.reshape((1, 1, 6, 7))

for i in range(10):
	with autograd.record():
		y_hat = conv2d(x)
		l = (y_hat - y) ** 2
	l.backward()

	conv2d.weight.data()[:] -= (0.03 * conv2d.weight.grad())
	print("batch %d, loss %.4f" % (i + 1, l.sum().asscalar()))

print(conv2d.weight.data().reshape((1, 2)))

# 互相关运算和卷积运算

# 特征图和感受野
