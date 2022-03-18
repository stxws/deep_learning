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

# from zero
def l2_penalty(w):
	return (w ** 2).sum() / 2

batch_size = 1
num_epochs = 100
lr = 0.003
train_data = g_data.ArrayDataset(train_features, train_labels)
train_iter = g_data.DataLoader(train_data, batch_size, shuffle=True)

def fit(lambd):
	w = nd.random.normal(scale=1, shape=(num_inputs, 1))
	b = nd.zeros(shape=(1,))
	train_loss = list()
	test_loss = list()
	for _ in range(num_epochs):
		for x, y in train_iter:
			w.attach_grad()
			b.attach_grad()
			with autograd.record():
				y_hat = nd.dot(x, w) + b
				loss = (y_hat - y) ** 2 / 2
				loss += lambd * l2_penalty(w)
			loss.backward()
			w = w - lr * w.grad / batch_size
			b = b - lr * b.grad / batch_size
		loss = (nd.dot(train_features, w) + b - train_labels) ** 2 / 2
		train_loss.append(loss.mean().asscalar())
		loss = (nd.dot(test_features, w) + b - test_labels) ** 2 / 2
		test_loss.append(loss.mean().asscalar())
	return train_loss, test_loss, w, b
epoch_list = [i + 1 for i in range(num_epochs)]

train_loss, test_loss, w, b = fit(lambd=0)
print("L2 normal of w: ", w.norm().asscalar())
plt.plot(epoch_list, train_loss, label="no_weight_decay_train")
plt.plot(epoch_list, test_loss, label="no_weight_decay_test")

train_loss, test_loss, w, b = fit(lambd=1)
print("L2 normal of w: ", w.norm().asscalar())
plt.plot(epoch_list, train_loss, label="lambd=1_train")
plt.plot(epoch_list, test_loss, label="lambd=1_test")

train_loss, test_loss, w, b = fit(lambd=3)
print("L2 normal of w: ", w.norm().asscalar())
plt.plot(epoch_list, train_loss, label="lambd=3_train")
plt.plot(epoch_list, test_loss, label="lambd=3_test")

plt.legend()
plt.show()
