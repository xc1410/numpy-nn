import time
import numpy as np
from mnist import load_mnist
from model import cross_entropy,init_params,forward,backward

train_images, train_labels, test_images, test_labels = load_mnist()

seed=0
lr=0.5
batch_size=100
epochs=10
steps=len(train_labels)//batch_size
losses=[]
accuracies=[]

np.random.seed(seed)
weight1, bias1, weight2, bias2 = init_params(784, 128, 10)
begin = time.perf_counter()

for epoch in range(epochs):
    order=np.random.permutation(len(train_labels))
    epoch_loss=0
    for step in range(steps):
        batch=order[step*batch_size:(step + 1) * batch_size]
        images = train_images[batch]
        labels = train_labels[batch]
        assert images.shape == (batch_size, 784), images.shape
        pre_relu,hidden,probs=forward(images, weight1, bias1, weight2, bias2)
        epoch_loss = epoch_loss + cross_entropy(probs, labels)
        grad_weight1, grad_bias1, grad_weight2, grad_bias2 = backward(
        images, labels, weight2, pre_relu, hidden, probs)
        weight1 = weight1 - lr * grad_weight1
        bias1 = bias1 - lr * grad_bias1
        weight2 = weight2 - lr * grad_weight2
        bias2 = bias2 - lr * grad_bias2
    pre_relu, hidden, probs = forward(test_images, weight1, bias1, weight2, bias2)
    accuracy = (probs.argmax(axis=1) == test_labels).mean()
    losses.append(epoch_loss / steps)
    accuracies.append(accuracy)
    print(epoch, epoch_loss / steps, accuracy)



print(time.perf_counter() - begin)
assert accuracy >= 0.96, accuracy


import matplotlib.pyplot as plt

plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), losses, marker="o")
plt.xlabel("epoch")
plt.ylabel("train loss")
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), accuracies, marker="o")
plt.xlabel("epoch")
plt.ylabel("test accuracy")
plt.tight_layout()
plt.savefig("curves.png")
print("saved curves.png")