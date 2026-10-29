import numpy as np
from mnist import load_mnist
from model import cross_entropy,init_params,forward

train_images,train_labels,test_images,test_labels=load_mnist()
images=train_images[:100]
labels=train_labels[:100]

np.random.seed(0)
weight1, bias1, weight2, bias2=init_params(784,128,10)
assert weight1.shape == (784, 128), weight1.shape
assert bias1.shape == (128,), bias1.shape
assert weight2.shape == (128, 10), weight2.shape
assert bias2.shape == (10,), bias2.shape

pre_relu,hidden,probs=forward(images,weight1,bias1,weight2,bias2)
print(images.shape, pre_relu.shape, hidden.shape, probs.shape)
assert pre_relu.shape == (100, 128), pre_relu.shape
assert hidden.shape == (100, 128), hidden.shape
assert probs.shape == (100, 10), probs.shape

print((pre_relu < 0).sum(), (hidden < 0).sum())
assert (hidden < 0).sum() == 0, (hidden < 0).sum()
assert np.allclose(probs.sum(axis=1), 1), probs.sum(axis=1)

loss = cross_entropy(probs, labels)
print(loss, np.log(10))
assert abs(loss - np.log(10)) < 0.01, loss
print((probs.argmax(axis=1) == labels).mean())
