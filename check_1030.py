import numpy as np
from model import cross_entropy,init_params,forward,backward

def backward_again(images, labels, weight2, pre_relu, hidden, probs):
    count=len(labels)
    onehot=np.zeros(probs.shape)
    onehot[np.arange(count),labels]=1
    grad_logits=(probs-onehot)/count
    grad_weight2=hidden.T@grad_logits
    grad_bias2=grad_logits.sum(axis=0)
    grad_hidden=grad_logits@weight2.T
    grad_pre_relu=grad_hidden*(pre_relu>0)
    grad_weight1=images.T@grad_pre_relu
    grad_bias1=grad_pre_relu.sum(axis=0)
    return grad_weight1, grad_bias1, grad_weight2, grad_bias2

np.random.seed(0)
images = np.random.randn(4, 6)
labels = np.array([0, 2, 1, 2])
weight1, bias1, weight2, bias2 = init_params(6, 5, 3)
pre_relu, hidden, probs = forward(images, weight1, bias1, weight2, bias2)
mine = backward_again(images, labels, weight2, pre_relu, hidden, probs)
reference = backward(images, labels, weight2, pre_relu, hidden, probs)
for mine_grad, reference_grad in zip(mine, reference):
    print(mine_grad.shape, np.allclose(mine_grad, reference_grad))
for mine_grad, reference_grad in zip(mine, reference):
    assert np.allclose(mine_grad, reference_grad), (mine_grad, reference_grad)

def compute_loss(weight1,bias1,weight2,bias2):
    pre_relu,hidden,probs=forward(images,weight1,bias1,weight2,bias2)
    return cross_entropy(probs,labels)

eps=1e-6
numeric_bias2=np.zeros((3,))
for col in range(3):
    plus=bias2.copy()
    plus[col]+=eps
    minus=bias2.copy()
    minus[col]-=eps
    numeric_bias2[col]=(compute_loss(weight1,bias1,weight2,plus)-compute_loss(weight1,bias1,weight2,minus))/(2*eps)

print(numeric_bias2)
print(reference[3])
assert np.allclose(numeric_bias2, reference[3]), (numeric_bias2, reference[3])