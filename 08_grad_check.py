import numpy as np
from model import cross_entropy,init_params,forward,backward

np.random.seed(0)
images=np.random.randn(4,6)
labels=np.array([0,2,1,2])
weight1,bias1,weight2,bias2=init_params(6,5,3)

def compute_loss(weight1,bias1,weight2,bias2):
    pre_relu,hidden,probs=forward(images,weight1,bias1,weight2,bias2)
    return cross_entropy(probs,labels)

pre_relu,hidden,probs=forward(images,weight1,bias1,weight2,bias2)
print(compute_loss(weight1, bias1, weight2, bias2),(pre_relu<=0).sum())
grad_weight1, grad_bias1, grad_weight2, grad_bias2 = backward(images, labels, weight2, pre_relu, hidden, probs)
assert grad_weight1.shape == (6, 5), grad_weight1.shape
assert grad_bias1.shape == (5,), grad_bias1.shape
assert grad_weight2.shape == (5, 3), grad_weight2.shape
assert grad_bias2.shape == (3,), grad_bias2.shape

eps=1e-6
numeric_weight1=np.zeros((6,5))
for row in range(6):
    for col in range(5):
        plus=weight1.copy()
        plus[row,col]+=eps
        minus = weight1.copy()
        minus[row,col]-=eps
        numeric_weight1[row,col]=(compute_loss(plus, bias1, weight2, bias2)- compute_loss(minus, bias1, weight2, bias2))/ (2 * eps)

numeric_weight2=np.zeros((5,3))
for row in range(5):
    for col in range(3):
        plus=weight2.copy()
        plus[row,col]+=eps
        minus = weight2.copy()
        minus[row,col]-=eps
        numeric_weight2[row,col]=(compute_loss(weight1, bias1,plus, bias2)- compute_loss(weight1, bias1, minus, bias2))/ (2 * eps)

numeric_bias1=np.zeros((5,))
for col in range(5):
    plus=bias1.copy()
    plus[col]+=eps
    minus = bias1.copy()
    minus[col]-=eps
    numeric_bias1[col]=(compute_loss(weight1, plus, weight2, bias2)- compute_loss(weight1, minus, weight2, bias2))/ (2 * eps)

numeric_bias2=np.zeros((3,))
for col in range(3):
    plus=bias2.copy()
    plus[col]+=eps
    minus = bias2.copy()
    minus[col]-=eps
    numeric_bias2[col]=(compute_loss(weight1, bias1, weight2, plus)- compute_loss(weight1, bias1, weight2, minus))/ (2 * eps)

print(numeric_weight1[0])
print(grad_weight1[0])
print(np.allclose(numeric_weight1, grad_weight1), np.allclose(numeric_bias1, grad_bias1), np.allclose(numeric_weight2, grad_weight2), np.allclose(numeric_bias2, grad_bias2))
assert np.allclose(numeric_weight1, grad_weight1), (numeric_weight1[0], grad_weight1[0])
assert np.allclose(numeric_bias1, grad_bias1), (numeric_bias1, grad_bias1)
assert np.allclose(numeric_weight2, grad_weight2), (numeric_weight2[0], grad_weight2[0])
assert np.allclose(numeric_bias2, grad_bias2), (numeric_bias2, grad_bias2)