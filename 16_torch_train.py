import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import load_mnist
from model import init_params
from torch_model import TwoLayerNet

train_images, train_labels, test_images, test_labels = load_mnist()
test_images_t = torch.tensor(test_images, dtype=torch.float64)
test_labels_t = torch.tensor(test_labels, dtype=torch.int64)

seed = 0
lr = 0.5
batch_size = 100
epochs = 10
steps = len(train_labels) // batch_size

np.random.seed(seed)
weight1, bias1, weight2, bias2 = init_params(784, 128, 10)
model = TwoLayerNet(784, 128, 10)
model.layer1.weight=nn.Parameter(torch.tensor(weight1.T))
model.layer1.bias=nn.Parameter(torch.tensor(bias1))
model.layer2.weight=nn.Parameter(torch.tensor(weight2.T))
model.layer2.bias=nn.Parameter(torch.tensor(bias2))
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
begin = time.perf_counter()
for epoch in range(epochs):
    order=np.random.permutation(len(train_labels))
    epoch_loss = 0
    for step in range(steps):
        batch=order[step*batch_size:(step+1)*batch_size]
        images_t=torch.tensor(train_images[batch],dtype=torch.float64)
        labels_t=torch.tensor(train_labels[batch],dtype=torch.int64)
        assert images_t.shape == (batch_size, 784), images_t.shape
        optimizer.zero_grad()
        logits_t=model(images_t)
        loss_t=F.cross_entropy(logits_t,labels_t)
        loss_t.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss_t.item()
    logits_t=model(test_images_t)
    accuracy=(logits_t.argmax(dim=1)==test_labels_t).sum().item()/len(test_labels)
    print(epoch, epoch_loss / steps, accuracy)
print(time.perf_counter() - begin)
assert accuracy >= 0.96, accuracy