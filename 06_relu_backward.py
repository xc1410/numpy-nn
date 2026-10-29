import numpy as np

pre_relu = np.array([[-1.0, 2.0], [3.0, -4.0]])
grad_hidden = np.array([[10.0, 20.0], [30.0, 40.0]])

grad_pre_relu = grad_hidden.copy()           # 先 .copy() 一份 grad_hidden
grad_pre_relu[pre_relu<=0] =0                        # pre_relu <= 0 的位置清零（x[mask] = 值）
print(grad_pre_relu)
assert np.allclose(grad_pre_relu, np.array([[0.0, 20.0], [30.0, 0.0]])), grad_pre_relu
assert grad_hidden[0, 0] == 10, grad_hidden

assert np.allclose(grad_hidden * (pre_relu > 0), grad_pre_relu), grad_hidden * (pre_relu > 0)