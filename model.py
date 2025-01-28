import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F

import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super(Network,self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    
    self.conv2 = nn.Conv2d(16,32,3,1,1)
    self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    
    self.conv3 = nn.Conv2d(32,64,3,1,1)
    self.pool3 = nn.MaxPool2d(2,2,0)
    
    self.conv4 = nn.Conv2d(64,128,3,1,1)
    self.pool4 = nn.MaxPool2d(2,2,0)
    
    self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    self.fc1 = nn.Linear(128*4*4,128)
    self.dropout1 = nn.Dropout(0.4)
    self.fc2 = nn.Linear(128,64)
    self.dropout2 = nn.Dropout(0.2)
    self.fc3 = nn.Linear(64,32)
    self.fc4 = nn.Linear(32,1)
    self.relu = nn.LeakyReLU(negative_slope=0.01)
    
  def forward(self,x):
    x = self.conv1(x) 
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool2(x)
    x = self.conv3(x)
    x = self.pool3(x)
    x = self.conv4(x)
    x = self.pool4(x)
    x = self.flatten(x)
    x = self.relu(self.fc1(x))
    x = self.dropout1(x)
    x = self.relu(self.fc2(x))
    x = self.dropout2(x)
    x = self.relu(self.fc3(x))
    x = self.fc4(x)
    
    return x
      
 
    

    
      

    