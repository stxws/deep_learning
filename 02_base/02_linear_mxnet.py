#!/usr/bin/python3

from mxnet import nd, autograd
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss
from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon
import numpy as np

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
dataset = g_data.ArrayDataset(features, labels)
data_iter = g_data.DataLoader(dataset, batch_size, shuffle=True)
# for x, y in data_iter:
# 	print(x, y)
# 	break

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(init.Normal(sigma=0.01))
loss = g_loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
num_epochs = 3
for epoch in range(1, num_epochs + 1):
	for x, y in data_iter:
		with autograd.record():
			l = loss(net(x), y)
		l.backward()
		trainer.step(batch_size)
	l = loss(net(features), labels)
	print("epoch %d, loss: %f" % (epoch, l.mean().asnumpy()))

dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())