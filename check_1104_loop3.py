import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist import load_mnist
from torch.utils.data import DataLoader,Dataset

class TwoLayerNet(nn.Module):
    def __init__(self,in_size,hidden_size,out_size):
        super().__init__()
        self.layer1=nn.Linear(in_size,hidden_size)
        self.layer2=nn.Linear(hidden_size,out_size)

    def forward(self,images_t):
        hidden_t=torch.relu(self.layer1(images_t))
        return self.layer2(hidden_t)

class ArrayData(Dataset):
    def __init__(self,inputs_t,labels_t):
        super().__init__()
        self.inputs_t=inputs_t
        self.labels_t=labels_t

    def __len__(self):
        return (len(self.labels_t))

    def __getitem__(self,index):
        return self.inputs_t[index],self.labels_t[index]


train_images,train_labels,test_images,test_labels=load_mnist()
test_images_t=torch.tensor(test_images,dtype=torch.float32)
test_labels_t=torch.tensor(test_labels,dtype=torch.int64)

train_data=ArrayData(
    torch.tensor(train_images,dtype=torch.float32),
    torch.tensor(train_labels,dtype=torch.int64)
)
batch_size=100
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
print(len(train_data),len(train_loader))

lr=0.5
epochs=10
model=TwoLayerNet(784,128,10)
optimizer=torch.optim.SGD(model.parameters(),lr=lr)
steps=len(train_labels)//batch_size

for epoch in range(epochs):
    epoch_loss=0
    for images_t,labels_t in train_loader:
        assert images_t.shape==(batch_size,784),images_t.shape
        optimizer.zero_grad()
        logits_t=model(images_t)
        loss_t=F.cross_entropy(logits_t,labels_t)
        loss_t.backward()
        optimizer.step()
        epoch_loss=epoch_loss+loss_t.item()
    logits_t=model(test_images_t)
    accuracy=((logits_t.argmax(dim=1)==test_labels_t).sum().item())/len(test_labels_t)
    print(epoch,epoch_loss/steps,accuracy)

assert accuracy>=0.96,accuracy