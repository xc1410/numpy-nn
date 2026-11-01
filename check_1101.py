import numpy as np
from mnist import load_mnist
from model import cross_entropy, init_params, forward, backward

train_images, train_labels, test_images, test_labels = load_mnist()
count = 1000
lr = 0.5
batch_size = 100
epochs = 3
steps = count // batch_size

np.random.seed(0)
weight1, bias1, weight2, bias2 = init_params(784, 128, 10)

for epoch in range(epochs):
    order=np.random.permutation(count)
    epoch_loss=0
    for step in range(steps):
        batch=order[step*batch_size:(step+1)*batch_size]
        images=train_images[batch]
        labels=train_labels[batch]
        pre_relu,hidden,probs=forward(images,weight1,bias1,weight2,bias2)
        epoch_loss=epoch_loss+cross_entropy(probs,labels)
        grad_weight1, grad_bias1, grad_weight2, grad_bias2 = backward(images, labels, weight2, pre_relu, hidden, probs)
        weight1=weight1-lr*grad_weight1
        bias1=bias1-lr*grad_bias1
        weight2=weight2-lr*grad_weight2
        bias2=bias2-lr*grad_bias2
    pre_relu,hidden,probs=forward(test_images,weight1,bias1,weight2,bias2)
    accuracy = (probs.argmax(axis=1) == test_labels).mean()
    print(epoch, epoch_loss / steps, accuracy)


assert accuracy > 0.7, accuracy