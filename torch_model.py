import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super().__init__()
        self.layer1=nn.Linear(in_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,out_size)

    def forward(self,images_t):
        x=torch.relu(self.layer1(images_t))
        return self.layer2(x)
    