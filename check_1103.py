import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import init_params

np.random.seed(0)
images=np.random.randn(4,6)
labels=np.array([0,2,1,2])
weight1,bias1,weight2,bias2=init_params(6,5,3)
lr=1.0

# ① 在这个文件里重写 class TwoLayerNet(nn.Module)，不 import torch_model
class TwoLayerNet(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super().__init__()
        self.layer1=nn.Linear(in_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,out_size)

    def forward(self,images_t):
        x=self.layer1(images_t)
        return self.layer2(torch.relu(x))
    
model=TwoLayerNet(6,5,3)
model.layer1.weight=nn.Parameter(torch.tensor(weight1.T))
model.layer1.bias=nn.Parameter(torch.tensor(bias1))
model.layer2.weight=nn.Parameter(torch.tensor(weight2.T))  
model.layer2.bias=nn.Parameter(torch.tensor(bias2))
optimizer=torch.optim.SGD(model.parameters(),lr=lr)
images_t=torch.tensor(images,dtype=torch.float64)
labels_t=torch.tensor(labels,dtype=torch.int64)


for step in range(3):
    optimizer.zero_grad()
    logits_t=model(images_t)
    loss_t=F.cross_entropy(logits_t, labels_t)
    loss_t.backward()
    optimizer.step()
    if step==0:
        first_loss = loss_t.item()
    print(step,loss_t.item())

assert loss_t.item() < first_loss, (first_loss, loss_t.item())
    