#!/usr/bin/python3

from mxnet import nd, gluon, autograd, init
from mxnet.gluon import nn
from mxnet.gluon import loss as g_loss
from mxnet.gluon import data as g_data
import matplotlib.pyplot as plt

def dropout(x, drop_prob):
	assert 0 <= drop_prob <= 1
	keep_prob = 1 - drop_prob
	if keep_prob == 0:
		return x.zeros_like()
	mask = nd.random.uniform(0, 1, x.shape) < keep_prob
	return mask * x / keep_prob

x = nd.arange(16).reshape((2, 8))
print(dropout(x, 0))
print(dropout(x, 0.5))
print(dropout(x, 1))
print("\n")

mnist_train = g_data.vision.FashionMNIST(train=True)
mnist_test = g_data.vision.FashionMNIST(train=False)
transformer = g_data.vision.transforms.ToTensor()
batch_size = 256
train_iter = g_data.DataLoader(mnist_train.transform_first(transformer), \
	batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = g_data.DataLoader(mnist_test.transform_first(transformer), \
	batch_size=batch_size, shuffle=False, num_workers=0)

num_inputs = 28 * 28
num_hiddens_1 = 256
drop_prob_1 = 0.2
num_hiddens_2 = 256
drop_prob_2 = 0.5
num_outputs = 10
w_1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens_1))
b_1 = nd.zeros(num_hiddens_1)
w_2 = nd.random.normal(scale=0.01, shape=(num_hiddens_1, num_hiddens_2))
b_2 = nd.zeros(num_hiddens_2)
w_3 = nd.random.normal(scale=0.01, shape=(num_hiddens_2, num_outputs))
b_3 = nd.zeros(num_outputs)
params = [w_1, b_1, w_2, b_2, w_3, b_3]

def softmax(x):
	x_exp = x.exp()
	partition = x_exp.sum(axis=0, keepdims=True)
	return x_exp / partition

def cross_entropy_loss(y_hat, y):
	return -1 * nd.pick(y_hat, y).log()

def net(x, params):
	[w_1, b_1, w_2, b_2, w_3, b_3] = params
	x = x.reshape((-1, num_inputs))
	hidden_1 = (nd.dot(x, w_1) + b_1).relu()
	if autograd.is_training():
		hidden_1 = dropout(hidden_1, drop_prob_1)
	hidden_2 = (nd.dot(hidden_1, w_2) + b_2).relu()
	if autograd.is_training():
		hidden_2 = dropout(hidden_2, drop_prob_2)
	out = nd.dot(hidden_2, w_2) + b_2
	return softmax(out)

# 训练
num_epochs = 20
lr = 0.5
for epoch in range(num_epochs):
	train_loss_sum = 0.0
	train_acc_sum = 0.0
	train_n = 0
	for x, y in train_iter:
		for param in params:
			param.attach_grad()
		with autograd.record():
			y_hat = net(x, params)
			loss = cross_entropy_loss(y_hat, y).sum()
		loss.backward()
		for i in range(len(params)):
			params[i] = params[i] - lr * params[i].grad / batch_size

		train_n += y.size
		train_loss_sum += loss.asscalar()
		train_acc_sum += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()

	test_acc_sum = 0.0
	test_n = 0
	for x, y in test_iter:
		test_acc_sum += (net(x, params).argmax(axis=1) == y.astype('float32')).sum().asscalar()
		test_n += y.size
	test_acc = test_acc_sum / test_n
	print('epoch %d, lr %.4f, loss %.4f, train acc %.3f, test acc %.3f' % \
		(epoch + 1, lr, train_loss_sum / train_n, train_acc_sum / train_n, test_acc))
	if epoch % 5 == 4:
		lr *= 0.5
