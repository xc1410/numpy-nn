import torch
import torch.nn as nn

layer= nn.Linear(2,1)
old_weight_t = layer.weight
layer.weight = nn.Parameter(torch.tensor([[1.0, 2.0]]))
layer.bias = nn.Parameter(torch.tensor([0.0]))
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

inputs_t = torch.tensor([[3.0, 4.0]])
loss_t = layer(inputs_t).sum()
loss_t.backward()
print(layer.weight.grad, old_weight_t.grad)
optimizer.step()
print(layer.weight)
assert layer.weight[0, 0].item() < 1.0, layer.weight