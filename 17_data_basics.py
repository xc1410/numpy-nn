import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_model import ArrayData

inputs_t = torch.tensor(np.arange(10).reshape(5, 2), dtype=torch.float64)
labels_t = torch.tensor([3, 1, 4, 1, 5])
data = ArrayData(inputs_t, labels_t)
print(len(data))
print(data[2])
loader = DataLoader(data, batch_size=2, shuffle=False)
print(len(loader))
for batch_inputs_t, batch_labels_t in loader:
    print(batch_inputs_t.shape, batch_labels_t)