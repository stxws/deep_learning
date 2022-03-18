#!/usr/bin/python3

from mxnet import init, nd, gluon
from mxnet.gluon import nn

# 读写NDArray
x = nd.ones(3)
nd.save("03_calculation/params/x", x)
x2 = nd.load("03_calculation/params/x")
print(x2)

y = nd.zeros(4)
nd.save("03_calculation/params/xy", [x, y])
x2, y2 = nd.load("03_calculation/params/xy")
print(x2, y2)

mydict = {"x": x, "y": y}
nd.save("03_calculation/params/mydict", mydict)
mydict2 = nd.load("03_calculation/params/mydict")
print(mydict2)


# 读写Gluon模型的参数
class MLP(nn.Block):
	def __init__(self, **kwargs):
		super(MLP, self).__init__(**kwargs)
		self.hidden = nn.Dense(256, activation="relu")
		self.output = nn.Dense(10)

	def forward(self, x):
		return self.output(self.hidden(x))

net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2, 10))
y = net(x)
print(y)
net.save_parameters("03_calculation/params/mlp.params")

net2 = MLP()
net2.load_parameters("03_calculation/params/mlp.params")
y2 = net2(x)
print(y2 == y)