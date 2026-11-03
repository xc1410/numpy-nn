import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from model import cross_entropy,init_params,forward,backward
from torch_model import TwoLayerNet

np.random.seed(0)
images = np.random.randn(4,6)
labels=np.array([0,2,1,2])
weight1, bias1, weight2, bias2 = init_params(6, 5, 3)
lr = 0.5

model = TwoLayerNet(6, 5, 3)
model.layer1.weight=nn.Parameter(torch.tensor(weight1.T))
model.layer1.bias=nn.Parameter(torch.tensor(bias1))
model.layer2.weight=nn.Parameter(torch.tensor(weight2.T))
model.layer2.bias=nn.Parameter(torch.tensor(bias2))
print(model.layer1.bias)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
images_t = torch.tensor(images)
labels_t = torch.tensor(labels)

for step in range(3):
    pre_relu, hidden, probs = forward(images, weight1, bias1, weight2, bias2)
    numpy_loss = cross_entropy(probs, labels)
    grad_weight1, grad_bias1, grad_weight2, grad_bias2 = backward(images, labels, weight2, pre_relu, hidden, probs)
    weight1 = weight1 - lr * grad_weight1
    bias1 = bias1 - lr * grad_bias1
    weight2 = weight2 - lr * grad_weight2
    bias2 = bias2 - lr * grad_bias2
    optimizer.zero_grad()
    logits_t=model(images_t)
    loss_t=F.cross_entropy(logits_t,labels_t)
    loss_t.backward()
    optimizer.step()
    print(step, loss_t.item(), np.allclose(loss_t.item(), numpy_loss))
    assert np.allclose(loss_t.item(), numpy_loss), (loss_t.item(), numpy_loss)