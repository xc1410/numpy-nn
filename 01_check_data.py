import numpy as np
from mnist import load_mnist
train_images,train_labels,test_images,test_labels=load_mnist()
print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
print(np.max(train_images),train_labels[:10])
print((train_labels==0).sum())
assert train_images.shape == (60000, 28), train_images.shape
assert test_images.shape == (10000, 784), test_images.shape