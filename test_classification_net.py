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
from antarcticnet import AntarcticNet



#### LOAD IN DIRECTORIES ####





train_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/train_subsets'
val_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/train_subsets'
model_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/models/'



#define batch size, num workers
batch_size = 7
num_workers = 0

# define transforms:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0),
                  transforms.ToTensor(), normalize])


test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize]) 


##### NET 1 - ROCKS



train_rock = "C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/txtdata/trainingfiles/trainingfile_rock_small.txt"
val_rock = "C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/txtdata/valfiles/valfile_rock.txt"

traintext = open(train_rock, "r")
valtext = open(val_rock, "r")


# define datasets:

train_data_rock = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_rock = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_rock = torch.utils.data.DataLoader(train_data_rock, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_rock = torch.utils.data.DataLoader(val_data_rock,  num_workers = 0, batch_size=batch_size)

                                                                                                                
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

#################################
#PREPPING THE MODEL FOR TRAINING#
#################################

# define model

rock_model = AntarcticNet()


# specify loss function
criterion = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = torch.optim.SGD(rock_model.parameters(), lr=0.001, momentum=0.09)

# specify scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

# save model
PATH = os.path.join(model_dir, 'rock_model.pt')
torch.save(rock_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_epochs = 10
n_iterations = int(len(train_data_rock)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



rock_model.train() # final step to prep model for training



for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    rock_model = torch.load(PATH)
    
    #prep to train
    rock_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_rock):  
        
        # extracting from dictionary 
        data = D['image']
        target = D['landmarks']
        #print(type(target))
        
        #target = target[0]
        target = target.view(-1, 1)
        

        # formatting and modifying output from dict
        input_img = data
        
        target = torch.tensor(target)
        #data = data.float()
                
        #### TRAINING PROPER ####
        #########################
        
        print("Epoch:", epoch + 1, "Iteration:", iter + 1, "out of:", n_iterations)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model     
        outputs = rock_model(input_img)
        
        # calculate the loss
        loss = criterion(outputs, target)
        print(loss.item())
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update running training loss
        train_loss += loss.item() #*input_img.size(0)
      
    #  scheduler - perform a its step in here - controls rate of learning
    scheduler.step()
    
    # print training statistics - calculate average loss over an epoch
    train_loss = train_loss/len(tl_rock.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    rock_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_rock):
                        
            #print(data)
            #print(target)
            
            # extracting from dictionary 
            data = D['image']
            target = D['landmarks']
            
            # test prints
            
            
            
            
            # formatting data from the dict
            target = torch.tensor(target)
            
            
            # VALIDATION PROPER
            
            # returns the output if < 1, else 1 - converts output to probability
            outputs = rock_model(data)

            
            #print(outputs)
            predicted, sp = torch.max(outputs.data, 1)
            
            presize = predicted.size()
            psize = list(presize)[0]
            
            for index in range(psize):
                entry = predicted[index]
                value = entry.item()
                if (value < 0):
                    value = 0.0
                else:
                    value = 1.0
                predicted[index] = torch.tensor(value)
            
            
            #print(outputs)
            #print(target)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()



    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(rock_model, PATH)


traintext.close()
valtext.close()
    
    
