import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from antarcticplotdataset_iterable import AntarcticPlotDataset



#### LOAD IN DIRECTORIES ####



txt_file_adr = "C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/contourdata_test.txt"
train_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/contours'


textfile = open(txt_file_adr, "r")


#### CONVERT DATA TO EXECUTABLE FORMAT ####

#define batch size, num workers
batch_size = 9
num_workers = 0

# define transforms:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



train_transform = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                  transforms.ToTensor(), normalize])


test_transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize]) 


# define datasets:

train_data = AntarcticPlotDataset(textfile, train_dir, transform=train_transform)
val_data = AntarcticPlotDataset(textfile, train_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

train_loader = torch.utils.data.DataLoader(train_data, num_workers = 0, batch_size=batch_size)

val_loader = torch.utils.data.DataLoader(val_data,  num_workers = 0, batch_size=batch_size)

                                                                                                                
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

#################################
#PREPPING THE MODEL FOR TRAINING#
#################################

# define model

model = models.squeezenet1_0(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=1)

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# specify scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# save model
PATH = "C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/model.pt"
torch.save(model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_epochs = 5
n_iterations = int(len(train_data)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



model.train() # final step to prep model for training



for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    model = torch.load(PATH)
    
    #prep to train
    model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(train_loader):  
        
        # extracting from dictionary 
        data = D['image']
        target = D['landmarks']
        

        # formatting and modifying output from dict
        input_img = data
        target = torch.tensor(target)

                
        #### TRAINING PROPER ####
        #########################
        
        print("Epoch:", epoch + 1, "Iteration:", iter + 1, "out of:", n_iterations)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model        
        outputs = model(input_img)
        
        # calculate the loss
        loss = criterion(outputs, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update running training loss
        train_loss += loss.item()*input_img.size(0)
      
    #  scheduler - perform a its step in here - controls rate of learning
    scheduler.step()
    
    # print training statistics - calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(val_loader):
            
            
            
            
            # extracting from dictionary 
            data = D['image']
            target = D['landmarks']
            
            # test prints
            print("im here")
            print(data)
            print(target)
            
            
            # formatting data from the dict
            target = torch.tensor(target)
            
            
            # VALIDATION PROPER
            
            # returns the output if < 1, else 1 - converts output to probability
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    #print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    PATH = "./model.pt"
    torch.save(model, PATH)
