#!/usr/bin/python3

from mxnet import init, nd, gluon
from mxnet.gluon import nn

# 不含模型参数的自定义层
class CenteredLayer(nn.Block):
	def __init__(self, **kwargs):
		super(CenteredLayer, self).__init__(**kwargs)
	
	def forward(self, x):
		return x - x.mean()

layer = CenteredLayer()
out = layer(nd.array([1, 2, 3, 4, 5]))
print(out)

net = nn.Sequential()
net.add(
	nn.Dense(128),
	CenteredLayer()
)
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
out = y.mean().asscalar()
print(out)
print()

# 含模型参数的自定义层
params = gluon.ParameterDict()
params.get("param2", shape=(2, 3))
print(params)

class MyDense(nn.Block):
	def __init__(self, units, in_units, **kwargs):
		super(MyDense, self).__init__(**kwargs)
		self.weight = self.params.get("weight", shape=(in_units, units))
		self.bias = self.params.get("bias", shape=(units,))
	
	def forward(self, x):
		linear = nd.dot(x, self.weight.data()) + self.bias.data()
		return nd.relu(linear)

dense = MyDense(units=3, in_units=5)
print(dense.params)
dense.initialize()
out = dense(nd.random.uniform(shape=(2, 5)))
print(out)

net = nn.Sequential()
net.add(
	MyDense(8, in_units=64),
	MyDense(1, in_units=8)
)
net.initialize()
out = net(nd.random.uniform(shape=(2, 64)))
print(out)

