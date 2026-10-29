import numpy as np

def softmax(logits):
    shifted = logits-np.max(logits,axis=1,keepdims=True)
    exps=np.exp(shifted)
    return exps/exps.sum(axis=1,keepdims=True)

def cross_entropy(probs,labels):
    picked=probs[np.arange(len(labels)),labels]
    return  (-np.log(picked)).mean()

np.random.seed(0)
images = np.random.randn(4, 3)          # 4 个样本，每个 3 个特征
labels = np.array([0, 2, 1, 2])
weight = np.random.randn(3, 5) * 0.1    # 5 个类别
bias = np.zeros(5)


def compute_loss(weight, bias):
    return cross_entropy(softmax(images @ weight + bias), labels)


def compute_grads(weight, bias):
    probs = softmax(images @ weight + bias)
    onehot = np.zeros((4, 5))
    onehot[np.arange(4), labels] = 1    # 成对下标赋值：每行正确类那一格设成 1
    grad_logits =      (probs-onehot)/4             # 你推的结果，一行
    grad_weight = images.T @ grad_logits
    grad_bias = grad_logits.sum(axis=0)
    return grad_weight, grad_bias


print(compute_loss(weight, bias))       # 期待 1.6199631373958499
grad_weight, grad_bias = compute_grads(weight, bias)
assert grad_weight.shape == (3, 5), grad_weight.shape
assert grad_bias.shape == (5,), grad_bias.shape

eps=1e-6
numeric_weight = np.zeros((3,5))
for row in range(3):
    for col in range(5):
        plus =weight.copy()
        plus[row, col] +=eps
        minus = weight.copy()
        minus[row, col] -= eps
        numeric_weight[row, col] = (compute_loss(plus, bias) - compute_loss(minus, bias)) / (2 * eps)

numeric_bias=np.zeros((5))
for col in range(5):
    plus =bias.copy()
    plus[col,] +=eps
    minus = bias.copy()
    minus[col,] -= eps
    numeric_bias[col,] = (compute_loss(weight, plus) - compute_loss(weight, minus)) / (2 * eps)

print(numeric_weight[0])
print(grad_weight[0])
assert np.allclose(numeric_weight, grad_weight), (numeric_weight[0], grad_weight[0])
assert np.allclose(numeric_bias, grad_bias), (numeric_bias, grad_bias)