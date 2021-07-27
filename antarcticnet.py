import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt


#specialized model for antarctic dataset


class AntarcticNet(nn.Module):
    
    
    def __init__(self):
        
        super(AntarcticNet, self).__init__()

        #planning for 3+1 stacks 
        
        #bias is default true
        
        self.conv1 = nn.Conv2d(3, 10, kernel_size = 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 16, kernel_size = 3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3,stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 44, kernel_size = 3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(44, 72, kernel_size = 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(72, 96, kernel_size = 3, stride=1, padding=1)
        
        #for stack 4
        
        self.conv7 = nn.Conv2d(96, 128, kernel_size = 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(128, 144, kernel_size = 3, stride =1, padding=1)
        
        #for stack 5
        
        self.conv9 = nn.Conv2d(144, 196, kernel_size = 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(196, 196, kernel_size = 3, stride =1, padding=1)
        
        
        # lin conversion
        
        
        self.lin = nn.Linear(in_features=25088, out_features=1024)
        self.lin2 = nn.Linear(in_features = 1024, out_features=128)
        self.lin3 = nn.Linear(in_features = 128, out_features=1)
        
        self.pool = nn.MaxPool2d(kernel_size = 3, stride=(2,2), padding=1)
        
    def forward(self, x):
        #stack 1

        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.relu(self.pool(x))
        
        #stack 2
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.relu(self.pool(x))
        
        #stack 3
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.relu(self.pool(x))

        #stack 4
        x = self.conv7(x)
        x = self.conv8(x)
        x =  self.pool(x)
        
        
        #stack 5
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool(x)
        x = torch.flatten(x)
        x = self.lin(x)
        x = self.lin2(x)
        return self.lin3(x)
    
    


model = AntarcticNet()
summary(model, (3, 244, 244))

        
        
        
        
