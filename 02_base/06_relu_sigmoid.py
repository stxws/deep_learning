#!/usr/bin/pyhton3

import matplotlib.pyplot as plt
from mxnet import nd, autograd

def xyplot(x, y, name):
	plt.plot(x.asnumpy(), y.asnumpy())
	plt.xlabel('x')
	plt.ylabel(name + '(x)')
	plt.show()


# relu
x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
	y = x.relu()
xyplot(x, y, 'relu')
y.backward()
xyplot(x, x.grad, 'grad of relu')


# sigmoid
x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
	y = x.sigmoid()
xyplot(x, y, 'sigmoid')
y.backward()
xyplot(x, x.grad, 'grad of sigmoid')
