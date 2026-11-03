import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import load_mnist
from torch_model import TwoLayerNet, ArrayData
from torch.utils.data import DataLoader

train_images, train_labels, test_images, test_labels = load_mnist()
test_images_t = torch.tensor(test_images, dtype=torch.float32)
test_labels_t = torch.tensor(test_labels, dtype=torch.int64)


lr = 0.5
batch_size = 100
epochs = 10
steps = len(train_labels) // batch_size
train_data = ArrayData(
    torch.tensor(train_images, dtype=torch.float32), 
    torch.tensor(train_labels, dtype=torch.int64))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
print(len(train_data), len(train_loader))


model = TwoLayerNet(784, 128, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
begin = time.perf_counter()
for epoch in range(epochs):
    epoch_loss = 0
    for images_t, labels_t in train_loader:
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