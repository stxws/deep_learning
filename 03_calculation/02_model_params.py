#!/usr/bin/python3

from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"))
net.add(nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
y = net(x)

# 访问模型参数
print(net[0].params)
print(type(net[0].params))
print(net[0].params["dense0_weight"])
print(net[0].weight)
print(net[0].weight.data())
print(net[0].weight.grad())

print(net[1].bias.data())
print(net.collect_params())
print(net.collect_params(".*weight"))

# 初始化模型参数
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])
net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[0].weight.data()[0])

# 自定义初始化方法
class MyInit(init.Initializer):
	def _init_weight(self, name, data):
		print("Init", name, data.shape)
		data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
		data *= (data.abs() >= 5)

net.initialize(init=MyInit(), force_reinit=True)
print(net[0].weight.data()[0])
net[0].weight.set_data(net[0].weight.data() + 1)
print(net[0].weight.data()[0])

# 共享模型参数
net = nn.Sequential()
shared = nn.Dense(8, activation="relu")
net.add(nn.Dense(8, activation="relu"),
		shared,
		nn.Dense(8, activation="relu", params=shared.params),
		nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)
print(net[1].weight.data()[0] == net[2].weight.data()[0])