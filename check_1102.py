import numpy as np
import torch
import torch.nn.functional as F
from model import softmax, cross_entropy

np.random.seed(0)
images = np.random.randn(4, 6)
labels = np.array([0, 2, 1, 2])
weight = np.random.randn(6, 3) * 0.01
bias = np.zeros(3)
count = len(labels)
# 闭卷：单层 softmax 回归
# ① numpy 手推：probs → onehot → grad_logits → grad_weight、grad_bias
logits=images@weight+bias
probs=softmax(logits)
onehot=np.zeros(probs.shape)
onehot[range(len(labels)),labels]=1
grad_logits=(probs-onehot)/count
grad_weight=images.T@grad_logits
grad_bias=grad_logits.sum(axis=0)
# ② torch：images_t、labels_t、weight_t、bias_t（后两个要梯度）→ logits_t → loss_t → print(loss_t) → backward
images_t=torch.tensor(images)
labels_t=torch.tensor(labels)
weight_t=torch.tensor(weight,requires_grad=True)
bias_t=torch.tensor(bias,requires_grad=True)
logits_t=images_t@weight_t+bias_t
loss_t=F.cross_entropy(logits_t,labels_t)
print(loss_t)
loss_t.backward()

# ③ print(np.allclose(grad_weight, weight_t.grad.numpy()), np.allclose(grad_bias, bias_t.grad.numpy()))
print(np.allclose(grad_weight, weight_t.grad.numpy()), np.allclose(grad_bias, bias_t.grad.numpy()))
# ④ 两条 assert，第二个参数写 (自己算的, torch 算的)，两个都是 numpy 数组
assert np.allclose(grad_weight, weight_t.grad.numpy()), (grad_weight,weight_t.grad.numpy())
assert np.allclose(grad_bias, bias_t.grad.numpy()), (grad_bias,bias_t.grad.numpy())