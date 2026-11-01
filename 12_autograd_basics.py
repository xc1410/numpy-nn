import torch

weight=torch.tensor(3.0,requires_grad=True)
loss=weight**2+2*weight
print(loss)
loss.backward()
print(weight.grad)
print(weight.item(), weight.grad.item())

loss = weight ** 2 + 2 * weight
loss.backward()
print(weight.grad)

weight.grad = None
loss = weight ** 2 + 2 * weight
loss.backward()
print(weight.grad)
assert weight.grad.item() == 8.0, weight.grad
