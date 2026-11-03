import numpy as np
import torch
import torch.nn as nn
from torch_model import TwoLayerNet

layer = nn.Linear(6,5)
print(layer.weight.shape,layer.bias.shape,layer.weight.dtype)
np.random.seed(0)
images=np.random.randn(4,6)
images_t=torch.tensor(images,dtype=torch.float32)
outputs_t = layer(images_t)
manual_t = images_t @ layer.weight.T + layer.bias
print(outputs_t.shape, ((outputs_t - manual_t) ** 2).sum().item() < 1e-10)

model = TwoLayerNet(6, 5, 3)
print(model)
print([param.shape for param in model.parameters()])
logits_t = model(images_t)
assert logits_t.shape == (4, 3), logits_t.shape
