import numpy as np
import torch
import torch.nn.functional as F
from model import cross_entropy, init_params, forward, backward

np.random.seed(0)
images = np.random.randn(4, 6)
labels = np.array([0, 2, 1, 2])
weight1, bias1, weight2, bias2 = init_params(6, 5, 3)
pre_relu, hidden, probs = forward(images, weight1, bias1, weight2, bias2)
grads = backward(images, labels, weight2, pre_relu, hidden, probs)
print(cross_entropy(probs, labels))




images_t=torch.tensor(images)
labels_t=torch.tensor(labels)

weight1_t=torch.tensor(weight1,requires_grad=True)
bias1_t=torch.tensor(bias1,requires_grad=True)
weight2_t=torch.tensor(weight2,requires_grad=True)
bias2_t=torch.tensor(bias2,requires_grad=True)
pre_relu_t=images_t@weight1_t+bias1_t
hidden_t=torch.relu(pre_relu_t)
logits_t=hidden_t@weight2_t+bias2_t
loss_t = F.cross_entropy(logits_t, labels_t)
print(loss_t)
print(np.allclose(loss_t.item(), cross_entropy(probs, labels)))
loss_t.backward()
torch_grads = [weight1_t.grad, bias1_t.grad, weight2_t.grad, bias2_t.grad]

for torch_grad, numpy_grad in zip(torch_grads, grads):
    print(torch_grad.shape, np.allclose(torch_grad.numpy(), numpy_grad))
for torch_grad, numpy_grad in zip(torch_grads, grads):
    assert np.allclose(torch_grad.numpy(), numpy_grad), (torch_grad.numpy(), numpy_grad)