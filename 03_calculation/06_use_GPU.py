#!/usr/bin/python3

import mxnet as mx
from mxnet import init, nd, gluon
from mxnet.gluon import nn

# 计算设备
print(mx.cpu(), mx.gpu(4), mx.gpu(5))

# NDArray的GPU计算
x = nd.array([1, 2, 3])
print(x)
print(x.context)

## GPU上的存储
a = nd.array([1, 2, 3], ctx=mx.gpu(4))
print(a)
b = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(5))
print(b)
y = x.copyto(mx.gpu(4))
print(y)
z = x.as_in_context(mx.gpu(4))
print(z)

print(y.as_in_context(mx.gpu(4)) is y)
print(y.copyto(mx.gpu(4)) is y)

## GPU上的计算
print((z + 2).exp() * y)

# Gluon的GPU计算
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu(4))
# net.initialize(ctx=mx.cpu())
print(net(y))
print(net[0].weight.data())