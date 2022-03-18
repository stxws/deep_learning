#!/usr/bin/python3

from mxnet import nd
from mxnet import autograd
from mxnet.gluon import loss as g_loss
from mxnet.gluon import data as g_data

mnist_train = g_data.vision.FashionMNIST(train=True)
mnist_test = g_data.vision.FashionMNIST(train=False)
transformer = g_data.vision.transforms.ToTensor()
batch_size = 256
train_iter = g_data.DataLoader(mnist_train.transform_first(transformer), \
	batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = g_data.DataLoader(mnist_test.transform_first(transformer), \
	batch_size=batch_size, shuffle=False, num_workers=0)


num_inputs = 28 * 28
num_hiddens = 256
num_outputs = 10
w_ih = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b_ih = nd.zeros(num_hiddens)
w_ho = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b_ho = nd.zeros(num_outputs)
params = [w_ih, b_ih, w_ho, b_ho]

# for param in params:
# 	param.attach_grad()

def relu(x):
	return nd.maximum(x, 0)

def net(x, params):
	x = x.reshape((-1, num_inputs))
	hid = relu(nd.dot(x, params[0]) + params[1])
	return nd.dot(hid, params[2]) + params[3]

loss = g_loss.SoftmaxCrossEntropyLoss()

def train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None):
	for epoch in range(num_epochs):
		train_loss_sum = 0.0
		train_acc_sum = 0.0
		train_n = 0
		for x, y in train_iter:
			for i in range(len(params)):
				params[i].attach_grad()
			with autograd.record():
				y_hat = net(x, params)
				l = loss(y_hat, y).sum()
			l.backward()
			for i in range(len(params)):
				params[i] =  params[i] - lr * params[i].grad / batch_size

			train_n += y.size
			train_loss_sum += l.asscalar()
			train_acc_sum += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
		
		test_acc_sum = 0.0
		test_n = 0
		for x, y in test_iter:
			test_acc_sum += (net(x, params).argmax(axis=1) == y.astype('float32')).sum().asscalar()
			test_n += y.size
		test_acc = test_acc_sum / test_n
		print('epoch %d, lr %.4f, loss %.4f, train acc %.3f, test acc %.3f' % \
			(epoch + 1, lr, train_loss_sum / train_n, train_acc_sum / train_n, test_acc))

num_epochs = 5
lr = 0.5
train(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)