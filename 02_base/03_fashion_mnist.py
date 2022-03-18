#!/usr/bin/python3

import sys
import time
import numpy as np
from mxnet.gluon import data as g_data

mnist_train = g_data.vision.FashionMNIST(train=True)
mnist_test = g_data.vision.FashionMNIST(train=False)
print(len(mnist_train), len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, feature.dtype)

def get_fashion_mnist_labels(labels):
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', \
		'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]
print(get_fashion_mnist_labels([label]))

batch_size = 256
transformer = g_data.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
	num_workers = 0
else:
	num_workers = 4
train_iter = g_data.DataLoader(mnist_train.transform_first(transformer), \
	batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = g_data.DataLoader(mnist_test.transform_first(transformer), \
	batch_size=batch_size, shuffle=False, num_workers=num_workers)

start_time = time.time()
for x, y in train_iter:
	continue
print("%.2f sec" % (time.time() - start_time))