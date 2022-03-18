#!/usr/bin/python3

from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
	def _init_weight(self, name, data):
		print("Init", name, data.shape)

# 延后初始化
net = nn.Sequential()
net.add(
	nn.Dense(256, activation="relu"),
	nn.Dense(10)
)
net.initialize(init=MyInit())

x = nd.random.uniform(shape=(2, 20))
y = net(x)
print()

y = net(x)


# 避免延后初始化
net.initialize(init=MyInit(), force_reinit=True)
print()

net = nn.Sequential()
net.add(
	nn.Dense(256, in_units=20, activation="relu"),
	nn.Dense(10, in_units=256)
)
net.initialize(init=MyInit())