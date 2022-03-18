#! /usr/bin/python3

from mxnet import nd
import numpy as np

#创建ndarray
x = nd.arange(12)
print(x)
print(x.shape)
print(x.size)

x = x.reshape((3, 4))
print(x)
print(x.shape)
print(x.size)

print(nd.zeros((2, 3, 4)))
print(nd.ones((3, 4)))
print(nd.random.normal(0, 1, shape=(3, 4)))

y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(y)


#运算
print(x + y)
print(x * y)
print(x / y)
print(y.exp())
print(nd.dot(x, y.T))
print(nd.concat(x, y, dim=0))
print(nd.concat(x, y, dim=1))
print(x == y)
print(x.sum())
print(x.norm())
print(x.norm().asscalar())


#广播机制
A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))
print(A, B)
print(A + B)


#索引
print(x)
print(x[1:3])
print(x[1:3, :])
print(x[1:3, 1:3])
print(x[:, 1:3])

x[1, 2] = 9
print(x)

x[1:2, :] = 12
print(x)

print()


#运算的内存开销
before = id(y)
y = y + x
print(id(y) == before)

z = y.zeros_like()
before = id(z)
z[:] = x + y
print(id(z) == before)
nd.elemwise_add(x, y, out=z)
print(id(z) == before)
z = x + y
print(id(z) == before)

before = id(z)
z += y
print(id(z) == before)


#NDArray和NumPy相互变换
p = np.ones((2, 3))

d = nd.array(p)
print(d)

p = d.asnumpy()
print(p)