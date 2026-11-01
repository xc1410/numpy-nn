import numpy as np
import torch

print(torch.__version__,torch.version.cuda,torch.cuda.is_available())
array = np.arange(6).reshape(2,3)*1.0
shared = torch.from_numpy(array)
copied = torch.tensor(array)
print(shared.dtype,copied.dtype,shared.shape)
array[0,0]=100
print(shared[0,0],copied[0,0])
back = copied.numpy()
print(type(back),back.shape,back.dtype)
small = torch.tensor([[1.0,2.0]])
print(small.dtype)
assert copied.shape == (2, 3), copied.shape