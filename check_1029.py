import numpy as np

def softmax(logits):
    shifted=logits-np.max(logits,axis=1,keepdims=True)
    exps=np.exp(shifted)
    return exps/exps.sum(axis=1,keepdims=True)

def cross_entopy(probs,labels):
    picked=probs[np.arange(len(labels)),labels]
    return (-np.log(picked).mean())

assert np.allclose(softmax(np.array([[1000.0, 1001.0, 1002.0]])).sum(axis=1),1),softmax(np.array([[1000.0, 1001.0, 1002.0]])).sum(axis=1)

assert np.allclose(cross_entopy(softmax(np.zeros((2, 3))),[0, 2]),np.log(3)),cross_entopy(softmax(np.zeros((2, 3))),[0, 2])

grad_hidden = np.array([[10.0, 20.0], [30.0, 40.0]])
pre_relu = np.array([[-1.0, 2.0], [3.0, -4.0]])
grad_pre_relu=grad_hidden.copy()
grad_pre_relu[pre_relu<=0]=0
assert np.allclose(grad_pre_relu,[[0.0, 20.0], [30.0, 0.0]]),grad_pre_relu
assert np.allclose(grad_hidden,[[10.0, 20.0], [30.0, 40.0]]),grad_hidden

