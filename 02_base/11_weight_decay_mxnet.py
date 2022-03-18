#!/usr/bin/python3

# higher dimensional linear regression

from mxnet import nd, gluon, autograd, init
from mxnet.gluon import nn
from mxnet.gluon import loss as g_loss
from mxnet.gluon import data as g_data
import matplotlib.pyplot as plt

n_train, n_test = 20, 100
num_inputs = 200
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

# generate data
features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

train_features = features[:n_train, :]
test_features = features[n_train:, :]
train_labels = labels[:n_train]
test_labels = labels[n_train:]

# mxnet
batch_size = 1
num_epochs = 100
lr = 0.003
train_data = g_data.ArrayDataset(train_features, train_labels)
train_iter = g_data.DataLoader(train_data, batch_size, shuffle=True)

def fit(wd):
	net = nn.Sequential()
	net.add(nn.Dense(1))
	net.initialize(init.Normal(sigma=1))
	loss = g_loss.L2Loss()
	trainer_w = gluon.Trainer(net.collect_params(".*weight"), 'sgd',
		{"learning_rate": lr, "wd": wd})
	trainer_b = gluon.Trainer(net.collect_params(".*bias"), 'sgd',
		{"learning_rate": lr})

	train_loss = list()
	test_loss = list()
	for _ in range(num_epochs):
		for x, y in train_iter:
			with autograd.record():
				y_hat = net(x)
				l = loss(y_hat, y)
			l.backward()
			trainer_w.step(batch_size)
			trainer_b.step(batch_size)
		l = loss(net(train_features), train_labels)
		train_loss.append(l.mean().asscalar())
		l = loss(net(test_features), test_labels)
		test_loss.append(l.mean().asscalar())
	return train_loss, test_loss, net[0].weight.data()
epoch_list = [i + 1 for i in range(num_epochs)]

train_loss, test_loss, w = fit(0)
print("L2 normal of w: ", w.norm().asscalar())
plt.plot(epoch_list, train_loss, label="no_weight_decay_train")
plt.plot(epoch_list, test_loss, label="no_weight_decay_test")

train_loss, test_loss, w = fit(1)
print("L2 normal of w: ", w.norm().asscalar())
plt.plot(epoch_list, train_loss, label="wd=1_train")
plt.plot(epoch_list, test_loss, label="wd=1_test")

train_loss, test_loss, w = fit(3)
print("L2 normal of w: ", w.norm().asscalar())
plt.plot(epoch_list, train_loss, label="wd=3_train")
plt.plot(epoch_list, test_loss, label="wd=3_test")

plt.legend()
plt.show()
