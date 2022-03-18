#!/usr/bin/python3

from mxnet import nd
from mxnet.gluon import nn

# 二维最大池化层和平均池化层
def pool2d(x, pool_size, mode='max'):
	p_h, p_w = pool_size
	y = nd.zeros((x.shape[0] - p_h + 1, x.shape[1] - p_w + 1))
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			if mode == 'max':
				y[i, j] = x[i: i + p_h, j: j + p_w].max()
			elif mode == 'avg':
				y[i, j] = x[i: i + p_h, j: j + p_w].mean()
	return y

x = nd.array([[0,1,2], [3,4,5], [6,7,8]])
y = pool2d(x, (2, 2))
print(y)
x = nd.array([[0,1,2], [3,4,5], [6,7,8]])
y = pool2d(x, (2, 2), 'avg')
print(y)

# 填充和步幅
x = nd.arange(16).reshape((1, 1, 4, 4))
print(x)
pool2d = nn.MaxPool2D(pool_size=(3, 3))
y = pool2d(x)
print(y)
pool2d = nn.MaxPool2D(pool_size=(3, 3), strides=2, padding=1)
y = pool2d(x)
print(y)
pool2d = nn.MaxPool2D(pool_size=(2, 3), strides=(2, 3), padding=(1, 2))
y = pool2d(x)
print(y)

# 多通道
x = nd.concat(x, x + 1, dim=1)
print(x)
pool2d = nn.MaxPool2D(pool_size=(3, 3), strides=2, padding=1)
y = pool2d(x)
print(y)