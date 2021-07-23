import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
from antarcticplotdataset_iterable import AntarcticPlotDataset



txt_file_adr = "C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/percent_cover_for_pytorch_set4.txt"

textfile = open(txt_file_adr, "r")



batch_size = 7
num_workers = 0

# define transforms:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                  transforms.ToTensor(), normalize])


test_transform = transforms.Compose([transforms.ToTensor(), normalize])


train_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/output_data_set4'


train_data = AntarcticPlotDataset(textfile, train_dir, transform=train_transform, start=0, end=27)
val_data = AntarcticPlotDataset(textfile, train_dir, transform=test_transform, start=0, end=27)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, num_workers = 0)

#val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle=True, num_workers = 0)

                                                                                                                
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)



model = models.squeezenet1_0(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=9)



# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# specify scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

PATH = "./model.pt"
torch.save(model, PATH)



# number of epochs to train the model
n_epochs = 5

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []
model.train() # prep model for training

n_iterations = int(len(train_data)/batch_size)

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    model = torch.load(PATH)
    model.train() # prep model for training
    ###################
    # train the model #
    ###################
    for iter, (data, target) in enumerate(train_loader):  
        
        
        
        print("Epoch:", epoch, "Iteration:", iter, "out of:", n_iterations)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(data)
        # calculate the loss
        loss = criterion(outputs, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update running training loss
        train_loss += loss.item()*data.size(0)
      
    # if you have a learning rate scheduler - perform a its step in here
    scheduler.step()
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

    # Run the test pass:
    correct = 0
    total = 0
    model.eval()  # prep model for validation

    with torch.no_grad():
        for data, target in val_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    #print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    PATH = "./model.pt"
    #change this
    torch.save(model, PATH)
