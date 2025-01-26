import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super(Network,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.conv2 = nn.Conv2d(16,32,3,1,1)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    self.fc1 = nn.Linear(32*16*16,128)
    self.relu = nn.ReLU(True)
    self.fc2 = nn.Linear(128,1)
    
  def forward(self,x):
   
    x = self.conv1(x) 
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.flatten(x)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x
      

    