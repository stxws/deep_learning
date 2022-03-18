#!/usr/bin/pyhton3

from mxnet import autograd, nd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon import data as g_data
from mxnet.gluon import loss as g_loss

mnist_train = g_data.vision.FashionMNIST(train=True)
mnist_test = g_data.vision.FashionMNIST(train=False)
transformer = g_data.vision.transforms.ToTensor()
batch_size = 256
train_iter = g_data.DataLoader(mnist_train.transform_first(transformer), \
	batch_size=batch_size, shuffle=True, num_workers=0)
test_iter = g_data.DataLoader(mnist_test.transform_first(transformer), \
	batch_size=batch_size, shuffle=False, num_workers=0)

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

loss = g_loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

num_epochs = 5
for epoch in range(num_epochs):
	train_loss_sum = 0.0
	train_acc_sum = 0.0
	train_n = 0
	for x, y in train_iter:
		with autograd.record():
			y_hat = net(x)
			l = loss(y_hat, y).sum()
		l.backward()
		trainer.step(batch_size)

		train_n += y.size
		train_loss_sum += l.asscalar()
		train_acc_sum += (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()
	
	test_acc_sum = 0.0
	test_n = 0
	for x, y in test_iter:
		test_acc_sum += (net(x).argmax(axis=1) == y.astype('float32')).sum().asscalar()
		test_n += y.size
	test_acc = test_acc_sum / test_n
	print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % \
		(epoch + 1, train_loss_sum / train_n, train_acc_sum / train_n, test_acc))