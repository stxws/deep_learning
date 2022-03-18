#!/usr/bin/python3

import matplotlib.pyplot as plt
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon import loss as g_loss
from mxnet.gluon import data as g_data

def train(train_features, test_features, train_labels, test_labels, loss, num_epochs, lr):
	net = nn.Sequential()
	net.add(nn.Dense(1))
	net.initialize()
	batch_size = min(20, train_labels.shape[0])
	train_iter = g_data.DataLoader( \
		g_data.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

	train_loss, test_loss = list(), list()
	for i in range(num_epochs):
		for x, y in train_iter:
			with autograd.record():
				l = loss(net(x), y)
			l.backward()
			trainer.step(batch_size)
		train_l = loss(net(train_features), train_labels).mean().asscalar()
		train_loss.append(train_l)
		test_l = loss(net(test_features), test_labels).mean().asscalar()
		test_loss.append(test_l)
	print('weight:', net[0].weight.data().asnumpy(),'\nbias:', net[0].bias.data().asnumpy())
	return train_loss, test_loss

loss = g_loss.L2Loss()
num_epochs = 40
lr = 0.01

n_train, n_test = 100, 100
true_w, true_b = [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3), dim=1)
labels = true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + \
	true_w[2] * poly_features[:, 2] + true_b
labels += nd.random.normal(scale=0.1, shape=labels.shape)

# 三阶多项式拟合（正常）
train_features = poly_features[:n_train, :]
train_labels = labels[:n_train]
test_features = poly_features[n_train:, :]
test_labels = labels[n_train:]
# print(features[:2], poly_features[:2], labels[:2])
train_loss, test_loss = train(train_features, test_features, train_labels, test_labels, loss, num_epochs, lr)
epoch_list = [i + 1 for i in range(num_epochs)]
plt.plot(epoch_list, train_loss, label="normal_train")
plt.plot(epoch_list, test_loss, label="normal_test")

# 线性函数拟合（欠拟合）
train_features = features[:n_train, :]
train_labels = labels[:n_train]
test_features = features[n_train:, :]
test_labels = labels[n_train:]
# print(features[:2], poly_features[:2], labels[:2])
train_loss, test_loss = train(train_features, test_features, train_labels, test_labels, loss, num_epochs, lr)
epoch_list = [i + 1 for i in range(num_epochs)]
plt.plot(epoch_list, train_loss, label="underfitting_train")
plt.plot(epoch_list, test_loss, label="underfitting_test")

# 训练样本不足（过拟合）
train_features = poly_features[:5, :]
train_labels = labels[:5]
test_features = poly_features[n_train:, :]
test_labels = labels[n_train:]
# print(features[:2], poly_features[:2], labels[:2])
train_loss, test_loss = train(train_features, test_features, train_labels, test_labels, loss, num_epochs, lr)
epoch_list = [i + 1 for i in range(num_epochs)]
plt.plot(epoch_list, train_loss, label="overfitting_train")
plt.plot(epoch_list, test_loss, label="overfitting_test")

plt.legend()
plt.show()