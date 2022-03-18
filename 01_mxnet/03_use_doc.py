##! /usr/bin/python3

from mxnet import nd
import numpy as np

print("\n\n dir")
print(dir(nd.random))
print(dir(np.random))

help(nd.ones_like)
help(np.ones_like)
