import mxnet as mx
from mxnet import autograd, nd, init, gluon
from mxnet.gluon import nn
from mxnet.gluon import utils as gutils
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata
import time

# 多GPU上初始化模型参数
class Residual(nn.Block):
	def __init__(self, num_clannels, strides=1, use_1x1conv=False, **kwargs):
		super(Residual, self).__init__(**kwargs)
		self.conv1 = nn.Conv2D(num_clannels, kernel_size=3, padding=1, 
								strides=strides)
		self.conv2 = nn.Conv2D(num_clannels, kernel_size=3, padding=1)
		if use_1x1conv:
			self.conv3 = nn.Conv2D(num_clannels, kernel_size=1, 
									strides=strides)
		else:
			self.conv3 = None
		self.bn1 = nn.BatchNorm()
		self.bn2 = nn.BatchNorm()
	
	def forward(self, x):
		y = self.conv1(x)
		y = self.bn1(y)
		y = nd.relu(y)
		y = self.conv2(y)
		y = self.bn2(y)
		if self.conv3:
			x = self.conv3(x)
		return nd.relu(x + y)

def resnet18(num_classes):
	def resnet_block(num_clannels, num_residuals, first_block=False):
		blk = nn.Sequential()
		for i in range(num_residuals):
			if i == 0 and not first_block:
				blk.add(Residual(num_clannels, strides=2, use_1x1conv=True))
			else:
				blk.add(Residual(num_clannels))
		return blk
	
	net = nn.Sequential()
	net.add(
		nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
		nn.BatchNorm(),
		nn.Activation('relu')
	)
	net.add(
		resnet_block(64, 2, first_block=True),
		resnet_block(128, 2),
		resnet_block(256, 2),
		resnet_block(256, 2)
	)
	net.add(
		nn.GlobalAvgPool2D(),
		nn.Dense(num_classes)
	)
	return net

net = resnet18(10)
ctxs = [mx.gpu(4), mx.gpu(5)]
net.initialize(init=init.Normal(sigma=0.01), ctx=ctxs)

x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gutils.split_and_load(x, ctxs)
gpu_y_hat = [net(gpu_x[0]), net(gpu_x[1])]
print(gpu_y_hat)

weight = net[0].params.get('weight')
try:
	weight.data()
except RuntimeError:
	print('not initialize on', mx.cpu())
print(weight.data(ctxs[0])[0])
print(weight.data(ctxs[1])[0])


# 多GPU训练模型
def load_data_fashion_mnist(batch_size):
	mnist_train = gdata.vision.FashionMNIST(train=True)
	mnist_test = gdata.vision.FashionMNIST(train=False)
	transformer = gdata.vision.transforms.ToTensor()
	batch_size = batch_size
	train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), 
		batch_size=batch_size, shuffle=True, num_workers=0)
	test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
		batch_size=batch_size, shuffle=False, num_workers=0)
	return train_iter, test_iter

def train(ctxs, batch_size, lr):
	trian_iter, test_iter = load_data_fashion_mnist(batch_size)
	print('running on:', ctxs)
	net.initialize(init=init.Normal(sigma=0.01), ctx=ctxs, force_reinit=True)
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {
		'learning_rate': lr
	})
	loss = gloss.SoftmaxCrossEntropyLoss()
	for epoch in range(4):
		start_time = time.time()
		for x, y in trian_iter:
			gpu_xs = gutils.split_and_load(x, ctxs)
			gpu_ys = gutils.split_and_load(y, ctxs)
			with autograd.record():
				ls = [loss(net(gpu_x), gpu_y)
						for gpu_x, gpu_y in zip(gpu_xs, gpu_ys)]
			for l in ls:
				l.backward()
			trainer.step(batch_size)
		nd.waitall()
		train_time = time.time() - start_time
		
		test_acc_sum, test_n = 0.0, 0
		for x, y in test_iter:
			x = x.as_in_context(ctxs[0])
			y = y.as_in_context(ctxs[0])
			# y_hat = lenet(x, gpu_params[0]).argmax(axis=1).astype("int32")
			y_hat = net(x)
			y_hat = y_hat.argmax(axis=1).astype("int32")
			test_acc_sum += (y_hat == y).sum().asscalar()
			test_n += y.size
		test_acc = test_acc_sum / test_n
		print("epoch %d, time %.1f sec, test acc %.3f" % \
			(epoch + 1, train_time, test_acc))

# ctxs = [mx.gpu(5)]
# train(ctxs, batch_size=256, lr=0.2)
ctxs = [mx.gpu(4), mx.gpu(5)]
train(ctxs, batch_size=512, lr=0.2)
