#!/usr/bin/python3

from mxnet import nd
from mxnet.gluon import nn

# 填充(padding)
def comp_conv2d(conv2d, x):
	conv2d.initialize()
	x = x.reshape((1, 1) + x.shape)
	y = conv2d(x)
	return y.reshape(y.shape[2:])

conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
x = nd.random.uniform(shape=(8, 8))
y = comp_conv2d(conv2d, x)
print(y.shape)

conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
y = comp_conv2d(conv2d, x)
print(y.shape)

# 步幅(stride)
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
y = comp_conv2d(conv2d, x)
print(y.shape)

conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
y = comp_conv2d(conv2d, x)
print(y.shape)
