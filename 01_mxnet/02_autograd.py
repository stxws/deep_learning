#! /usr/bin/python3

from mxnet import autograd, nd

x = nd.arange(4).reshape((4, 1))
print(x)

# 自动求梯度
x.attach_grad()
with autograd.record():
	y = 2 * nd.dot(x.T, x)
y.backward()
print(x.grad)


# 训练模式和预测模式
print(autograd.is_training())
with autograd.record():
	print(autograd.is_training())


# 对Python控制流求梯度
def fun(a):
	b = a * 2
	while b.norm().asscalar() < 1000:
		b = b * 2
	if b.sum().asscalar() > 0:
		c = b
	else:
		c = 100 * b
	return c

a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
	c = fun(a)
c.backward()
print(a.grad == c / a)