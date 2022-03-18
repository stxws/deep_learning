import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata
import time

# 数据并行

# 定义模型
scale = 0.01
w1 = nd.random.normal(scale=scale, shape=(20, 1, 3, 3))
b1 = nd.zeros(shape=20)
w2 = nd.random.normal(scale=scale, shape=(50, 20, 5, 5))
b2 = nd.zeros(shape=50)
w3 = nd.random.normal(scale=scale, shape=(800, 128))
b3 = nd.zeros(shape=128)
w4 = nd.random.normal(scale=scale, shape=(128, 10))
b4 = nd.zeros(shape=10)
params = [w1, b1, w2, b2, w3, b3, w4, b4]

def lenet(x, params):
	h1_conv = nd.Convolution(data=x, weight=params[0], bias=params[1],
							 kernel=(3, 3), num_filter=20)
	h1_activation = nd.relu(h1_conv)
	h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2, 2),
					stride=(2, 2))
	
	h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3],
							 kernel=(5, 5), num_filter=50)
	h2_activate = nd.relu(h2_conv)
	h2 = nd.Pooling(data=h2_activate, pool_type="avg", kernel=(2, 2),
					stride=(2, 2))
	h2 = nd.flatten(h2)

	h3_linear = nd.dot(h2, params[4] + params[5])
	h3 = nd.relu(h3_linear)
	y_hat = nd.dot(h3, params[6] + params[7])

	return y_hat

loss = gloss.SoftmaxCrossEntropyLoss()


# 多GPU之间同步数据
def get_params(params, ctx):
	new_params = [p.copyto(mx.gpu(4)) for p in params]
	for p in new_params:
		p.attach_grad()
	return new_params

new_params = [p.copyto(mx.gpu(4)) for p in params]
for p in new_params:
	p.attach_grad()
print("b1 weight:", new_params[1])
print("b1 grad:", new_params[1].grad)

def allreduce(data):
	for i in range(1, len(data)):
		data[0][:] += data[i].copyto(data[0].context)
	for i in range(1, len(data)):
		data[0].copyto(data[i])

data = [nd.ones(shape=(1, 2), ctx=mx.gpu(i)) * (i + 1) for i in (4, 5)]
print("before allreduce:", data)
allreduce(data)
print("after allreduce:", data)

def split_and_load(data, ctxs):
	n, k = data.shape[0], len(ctxs)
	m = n // k
	assert m * k == n, "# examples is not divided by # devices."
	return [data[c_i * m: (c_i + 1) * m].as_in_context(ctx) 
			for c_i, ctx in enumerate(ctxs)]

batch = nd.arange(24).reshape((6, 4))
ctxs = [mx.gpu(4), mx.gpu(5)]
splitted = split_and_load(batch, ctxs)
print("input: ", batch)
print("load into", ctxs)
print("output: ", splitted)


# 单个小批量上的多GPU训练
def train_batch(x, y, net, gpu_params, loss, lr, ctxs):
	gpu_xs = split_and_load(x, ctxs)
	gpu_ys = split_and_load(y, ctxs)
	ls = []
	for c_i in range(len(ctxs)):
		for p_i in range(len(gpu_params[0])):
			gpu_params[c_i][p_i].attach_grad()
		with autograd.record():
			ls.append(loss(net(gpu_xs[c_i], gpu_params[c_i]), gpu_ys[c_i]))
	for l in ls:
		l.backward()
	for i in range(len(gpu_params[0])):
		allreduce([gpu_params[c_i][i].grad for c_i in range(len(ctxs))])
	for c_i in range(len(ctxs)):
		for p_i in range(len(gpu_params[0])):
			gpu_params[c_i][p_i] = gpu_params[c_i][p_i] - lr * gpu_params[c_i][p_i].grad / x.shape[0]
	return gpu_params


# 定义训练函数
def load_data_fashion_mnist(batch_size):
	mnist_train = gdata.vision.FashionMNIST(train=True)
	mnist_test = gdata.vision.FashionMNIST(train=False)
	transformer = gdata.vision.transforms.ToTensor()
	batch_size = batch_size
	train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), 	batch_size=batch_size, shuffle=True, num_workers=0)
	test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
		batch_size=batch_size, shuffle=False, num_workers=0)
	return train_iter, test_iter

def train(batch_size, lr, ctxs):
	train_iter, test_iter = load_data_fashion_mnist(batch_size)
	gpu_params = []
	for ctx in ctxs:
		print("run on :", ctx)
		gpu_params.append([p.copyto(ctx) for p in params])
	for epoch in range(4):
		start_time = time.time()
		for x, y in train_iter:
			gpu_params = train_batch(x, y, lenet, gpu_params, loss, lr, ctxs)
			nd.waitall()
		train_time = time.time() - start_time

		test_acc_sum, test_n = 0.0, 0
		for x, y in test_iter:
			x = x.as_in_context(ctxs[0])
			y = y.as_in_context(ctxs[0])
			# y_hat = lenet(x, gpu_params[0]).argmax(axis=1).astype("int32")
			y_hat = lenet(x, gpu_params[0])
			y_hat = y_hat.argmax(axis=1).astype("int32")
			test_acc_sum += (y_hat == y).sum().asscalar()
			test_n += y.size
		test_acc = test_acc_sum / test_n
		print("epoch %d, time %.1f sec, test acc %.3f" % \
			(epoch + 1, train_time, test_acc))


# 多GPU训练实验
ctxs = [mx.gpu(4), mx.gpu(5)]
train(batch_size=256, lr=0.1, ctxs=ctxs) # 训练后网络输出为全0，原因还没找到
