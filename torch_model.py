import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TwoLayerNet(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super().__init__()
        self.layer1=nn.Linear(in_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,out_size)

    def forward(self,images_t):
        x=torch.relu(self.layer1(images_t))
        return self.layer2(x)

class ArrayData(Dataset):
    def __init__(self,inputs_t,labels_t):
        super().__init__()
        self.inputs_t=inputs_t
        self.labels_t=labels_t

    def __len__(self):
        return len(self.labels_t)

    def __getitem__(self, index):
        return self.inputs_t[index], self.labels_t[index]