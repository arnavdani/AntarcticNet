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
        
        self.conv3 = nn.Conv2d(16, 24, kernel_size = 3,stride=1, padding=1)
        self.conv4 = nn.Conv2d(24, 32, kernel_size = 3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(32, 44, kernel_size = 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(44, 64, kernel_size = 3, stride=1, padding=1)
        
        #for stack 4
        
        self.conv7 = nn.Conv2d(64, 81, kernel_size = 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(81, 96, kernel_size = 3, stride =1, padding=1)
        
        #for stack 5
        
        self.conv9 = nn.Conv2d(96, 128, kernel_size = 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size = 3, stride =1, padding=1)
        
        
        # lin conversion
        
        
        self.lin = nn.Linear(in_features=6272, out_features=128)
        self.lin2 = nn.Linear(in_features = 128, out_features=1)
        
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
        x = torch.relu(self.pool(x))
        
        
        
        #stack 5
        
        x = self.conv9(x)
        x = torch.relu(self.pool(x)) #last pooling layer
        
        
        x = torch.flatten(x, start_dim=1)
        
        
        
        x = self.lin(x)
        return self.lin2(x)
    
    


model = AntarcticNet()
print(model)
#summary(model, (3, 224, 224))

        
        
        
        
