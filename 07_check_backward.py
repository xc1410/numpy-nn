import numpy as np
from mnist import load_mnist
from model import cross_entropy, init_params, forward, backward

train_images, train_labels, test_images, test_labels = load_mnist()
images = train_images[:100]
labels = train_labels[:100]

np.random.seed(0)
weight1, bias1, weight2, bias2 = init_params(784, 128, 10)
pre_relu, hidden, probs = forward(images, weight1, bias1, weight2, bias2)
grad_weight1, grad_bias1, grad_weight2, grad_bias2 = backward(images, labels, weight2, pre_relu, hidden, probs)
assert grad_weight1.shape == weight1.shape, grad_weight1.shape
assert grad_bias1.shape == bias1.shape, grad_bias1.shape
assert grad_weight2.shape == weight2.shape, grad_weight2.shape
assert grad_bias2.shape == bias2.shape, grad_bias2.shape

lr=0.5
before=cross_entropy(probs,labels)
weight1-=lr*grad_weight1
bias1-=lr*grad_bias1
weight2-=lr*grad_weight2
bias2-=lr*grad_bias2
pre_relu,hidden,probs=forward(images,weight1,bias1,weight2,bias2)
after=cross_entropy(probs,labels)
print(before,after)
assert after<before,(before,after)