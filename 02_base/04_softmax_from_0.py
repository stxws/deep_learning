#!/usr/bin/pyhton3

from mxnet import autograd, nd
from mxnet.gluon import data as g_data

# x = nd.array([[1, 2, 3], [4, 5, 6]])
# print(x.sum(axis=0, keepdims=True), x.sum(axis=1, keepdims=True))

# y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = nd.array([0, 2], dtype='int32')
# print(nd.pick(y_hat, y))

# def accuracy(y_hat, y):
# 	result = y_hat.argmax(axis=1).astype('int32')
# 	result = (result == y)
# 	return result.astype('float32').mean().asscalar()

# print(accuracy(y_hat, y))


mnist_train = g_data.vision.FashionMNIST(train=True)
mnist_test = g_data.vision.FashionMNIST(train=False)
transformer = g_data.vision.transforms.ToTensor()
batch_size = 256
train_iter = g_data.DataLoader(mnist_train.transform_first(transformer), \
	batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = g_data.DataLoader(mnist_test.transform_first(transformer), \
	batch_size=batch_size, shuffle=False, num_workers=0)

num_inputs = 28 * 28
num_outputs = 10
w = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

def softmax(x):
	x_exp = x.exp()
	partition = x_exp.sum(axis=0, keepdims=True)
	return x_exp / partition

def cross_entropy_loss(y_hat, y):
	return -1 * nd.pick(y_hat, y).log()

def net(x):
	x = x.reshape((-1, num_inputs))
	return softmax(nd.dot(x, w) + b)

# x = nd.random.normal(shape=(2, 5))
# x_prob = softmax(x)
# print(x_prob, x_prob.sum(axis=0))

# 训练
num_epochs = 5
lr = 0.1
for epoch in range(num_epochs):
	train_loss_sum = 0.0
	train_acc_sum = 0.0
	train_n = 0
	for x, y in train_iter:
		w.attach_grad()
		b.attach_grad()
		with autograd.record():
			y_hat = net(x)
			loss = cross_entropy_loss(y_hat, y).sum()
		loss.backward()
		w = w - lr * w.grad / batch_size
		b = b - lr * b.grad / batch_size

		train_n += y.size
		train_loss_sum += loss.asscalar()
		train_acc_sum += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
	
	test_acc_sum = 0.0
	test_n = 0
	for x, y in test_iter:
		test_acc_sum += (net(x).argmax(axis=1) == y.astype('float32')).sum().asscalar()
		test_n += y.size
	test_acc = test_acc_sum / test_n
	print('epoch %d, lr %.4f, loss %.4f, train acc %.3f, test acc %.3f' % \
		(epoch + 1, lr, train_loss_sum / train_n, train_acc_sum / train_n, test_acc))