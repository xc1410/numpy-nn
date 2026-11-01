import numpy as np
from mnist import load_mnist
from model import softmax, cross_entropy

train_images, train_labels, test_images, test_labels = load_mnist()

lr = 0.5
batch_size = 100
epochs = 10
steps = len(train_labels) // batch_size

np.random.seed(0)
weight = np.random.randn(784, 10) * 0.01
bias = np.zeros(10)

for epoch in range(epochs):
    order=np.random.permutation(len(train_labels))
    epoch_loss=0
    for step in range(steps):
        batch=order[step*batch_size:(step+1)*batch_size]
        images=train_images[batch]
        labels=train_labels[batch]
        probs = softmax(images @ weight + bias)
        assert probs.shape == (batch_size, 10), probs.shape
        epoch_loss = epoch_loss + cross_entropy(probs, labels)
        onehot = np.zeros((len(labels), 10))
        onehot[np.arange(len(labels)),labels]=1
        grad_logits=(probs-onehot)/batch_size
        grad_weight=images.T@grad_logits
        grad_bias=grad_logits.sum(axis=0)
        weight=weight-lr*grad_weight
        bias=bias-lr*grad_bias
    test_probs = softmax(test_images @ weight + bias)
    predictions = test_probs.argmax(axis=1)
    accuracy = (predictions == test_labels).mean()
    print(epoch, epoch_loss / steps, accuracy)