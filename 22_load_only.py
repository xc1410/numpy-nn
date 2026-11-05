import time
import torch
from torch.utils.data import DataLoader
from mnist import load_mnist
from torch_model import ArrayData

train_images, train_labels, test_images, test_labels = load_mnist()
batch_size = 100
epochs = 10
train_data = ArrayData(
    torch.tensor(train_images, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.int64))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

count = 0
begin = time.perf_counter()
for epoch in range(epochs):
    for images_t, labels_t in train_loader:
        count = count + len(labels_t)
seconds = time.perf_counter() - begin
print(count, seconds, count / seconds)
assert count == len(train_data) * epochs, count