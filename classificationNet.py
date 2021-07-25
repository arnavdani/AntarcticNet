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





train_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/train_subsets'
val_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/train_subsets'
model_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/models/'
tf_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/txtdata/trainingfiles/trainingfile_'
vf_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/txtdata/valfiles/valfile_'



#define batch size, num workers
batch_size = 64
num_workers = 0

# define transforms:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])



train_transform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
                  transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0),
                  transforms.ToTensor(), normalize])


test_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize]) 


##### NET 1 - ROCKS



train_rock = os.path.join(tf_dir, 'rock.txt')
val_rock = os.path.join(vf_dir, 'rock.txt')

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

rock_model = models.resnet18(pretrained=False)
rock_model.fc = nn.Linear(in_features=512, out_features=1)

# specify loss function
criterion = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = torch.optim.SGD(rock_model.parameters(), lr=0.001, momentum=0.95)

# specify scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)

# save model
PATH = os.path.join(model_dir, 'rock_model.pt')
torch.save(rock_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_epochs = 5
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
        

        # formatting and modifying output from dict
        input_img = data
        target = torch.tensor(target)

                
        #### TRAINING PROPER ####
        #########################
        
        print("Epoch:", epoch + 1, "Iteration:", iter + 1, "out of:", n_iterations)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        
        # forward pass: compute predicted outputs by passing inputs to the model        
        outputs = rock_model(input_img)
        
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
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(rock_model, PATH)


traintext.close()
valtext.close()
    
    
    
    
    



#####################################################################################################################



#soil model


train_soil = os.path.join(tf_dir, 'soil.txt')
val_soil = os.path.join(vf_dir, 'soil.txt')

traintext = open(train_soil, "r")
valtext = open(val_soil, "r")


# define datasets:

train_data_soil = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_soil = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_soil = torch.utils.data.DataLoader(train_data_soil, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_soil = torch.utils.data.DataLoader(val_data_soil,  num_workers = 0, batch_size=batch_size)


# define model
soil_model = models.resnet18(pretrained=False)
soil_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'soil_model.pt')
torch.save(soil_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_soil)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



soil_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    soil_model = torch.load(PATH)
    
    #prep to train
    soil_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_soil):  
        
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
        outputs = soil_model(input_img)
        
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
    train_loss = train_loss/len(tl_soil.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    soil_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_soil):
                        
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
            outputs = soil_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(soil_model, PATH)


traintext.close()
valtext.close()




#####################################################################################################################



#white lichen model



train_WL = os.path.join(tf_dir, 'whli.txt')
val_WL = os.path.join(vf_dir, 'whli.txt')

traintext = open(train_WL, "r")
valtext = open(val_WL, "r")


# define datasets:

train_data_WL = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_WL = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_WL = torch.utils.data.DataLoader(train_data_WL, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_WL = torch.utils.data.DataLoader(val_data_WL,  num_workers = 0, batch_size=batch_size)


# define model
WL_model = models.resnet18(pretrained=False)
WL_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'wlichen_model.pt')
torch.save(WL_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_WL)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



WL_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    WL_model = torch.load(PATH)
    
    #prep to train
    WL_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_WL):  
        
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
        outputs = WL_model(input_img)
        
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
    train_loss = train_loss/len(tl_WL.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    WL_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_WL):
                        
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
            outputs = WL_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(WL_model, PATH)


traintext.close()
valtext.close()



#####################################################################################################################


#Bryum Spo/dead moss/other category


train_bry = os.path.join(tf_dir, 'rand.txt')
val_bry = os.path.join(vf_dir, 'rand.txt')

traintext = open(train_bry, "r")
valtext = open(val_bry, "r")


# define datasets:

train_data_bry = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_bry = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_bry = torch.utils.data.DataLoader(train_data_bry, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_bry = torch.utils.data.DataLoader(val_data_bry,  num_workers = 0, batch_size=batch_size)


# define model
bryum_model = models.resnet18(pretrained=False)
bryum_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'bryum_model.pt')
torch.save(bryum_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_bry)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



bryum_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    bryum_model = torch.load(PATH)
    
    #prep to train
    bryum_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_bry):  
        
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
        outputs = bryum_model(input_img)
        
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
    train_loss = train_loss/len(tl_bry.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    bryum_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_bry):
                        
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
            outputs = bryum_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(bryum_model, PATH)


traintext.close()
valtext.close()




#####################################################################################################################


#sanionia/brown moss model



train_san = os.path.join(tf_dir, 'brmo.txt')
val_san = os.path.join(vf_dir, 'brmo.txt')

traintext = open(train_san, "r")
valtext = open(val_san, "r")


# define datasets:

train_data_san = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_san = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_san = torch.utils.data.DataLoader(train_data_san, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_san = torch.utils.data.DataLoader(val_data_san,  num_workers = 0, batch_size=batch_size)


# define model
sanionia_model = models.resnet18(pretrained=False)
sanionia_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'sanionia_model.pt')
torch.save(sanionia_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_san)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



sanionia_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    sanionia_model = torch.load(PATH)
    
    #prep to train
    sanionia_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_san):  
        
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
        outputs = sanionia_model(input_img)
        
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
    train_loss = train_loss/len(tl_san.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    sanionia_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_san):
                        
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
            outputs = sanionia_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(sanionia_model, PATH)


traintext.close()
valtext.close()



#####################################################################################################################


#hairgrass model


train_grass = os.path.join(tf_dir, 'gras.txt')
val_grass = os.path.join(vf_dir, 'gras.txt')

traintext = open(train_grass, "r")
valtext = open(val_grass, "r")


# define datasets:

train_data_grass = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_grass = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_grass = torch.utils.data.DataLoader(train_data_grass, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_grass = torch.utils.data.DataLoader(val_data_grass,  num_workers = 0, batch_size=batch_size)


# define model
grass_model = models.resnet18(pretrained=False)
grass_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'grass_model.pt')
torch.save(grass_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_grass)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



grass_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    grass_model = torch.load(PATH)
    
    #prep to train
    grass_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_grass):  
        
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
        outputs = grass_model(input_img)
        
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
    train_loss = train_loss/len(tl_grass.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    grass_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_grass):
                        
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
            outputs = grass_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(grass_model, PATH)


traintext.close()
valtext.close()


#####################################################################################################################


#model for POLYTRICHIUM STRICTUM

#this is the green spiky moss thing

#abbr to PS, model name polytrich_model


train_PS = os.path.join(tf_dir, 'pomo.txt')
val_PS = os.path.join(vf_dir, 'pomo.txt')

traintext = open(train_PS, "r")
valtext = open(val_PS, "r")


# define datasets:

train_data_PS = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_PS = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_PS = torch.utils.data.DataLoader(train_data_PS, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_PS = torch.utils.data.DataLoader(val_data_PS,  num_workers = 0, batch_size=batch_size)


# define model
polytrich_model = models.resnet18(pretrained=False)
polytrich_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'polytrich_model.pt')
torch.save(polytrich_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_PS)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



polytrich_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    polytrich_model = torch.load(PATH)
    
    #prep to train
    polytrich_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_PS):  
        
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
        outputs = polytrich_model(input_img)
        
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
    train_loss = train_loss/len(tl_PS.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    polytrich_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_PS):
                        
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
            outputs = polytrich_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(polytrich_model, PATH)


traintext.close()
valtext.close()


#####################################################################################################################


#model for CHORISODONTIUM ACIPHYLLUM

#this is the light green smooth moss thing

#abbr to chor, model name chor_model


train_chor = os.path.join(tf_dir, 'chmo.txt')
val_chor = os.path.join(vf_dir, 'chmo.txt')

traintext = open(train_chor, "r")
valtext = open(val_chor, "r")


# define datasets:

train_data_chor = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_chor = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_chor = torch.utils.data.DataLoader(train_data_chor, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_chor = torch.utils.data.DataLoader(val_data_chor,  num_workers = 0, batch_size=batch_size)


# define model
chor_model = models.resnet18(pretrained=False)
chor_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'chor_model.pt')
torch.save(chor_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_chor)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



chor_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    chor_model = torch.load(PATH)
    
    #prep to train
    chor_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_chor):  
        
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
        outputs = chor_model(input_img)
        
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
    train_loss = train_loss/len(tl_chor.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    chor_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_chor):
                        
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
            outputs = chor_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(chor_model, PATH)


traintext.close()
valtext.close()




#####################################################################################################################


#brown lichen, abbr blichen


train_blichen = os.path.join(tf_dir, 'brli.txt')
val_blichen = os.path.join(vf_dir, 'brli.txt')

traintext = open(train_blichen, "r")
valtext = open(val_blichen, "r")


# define datasets:

train_data_blichen = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_blichen = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_blichen = torch.utils.data.DataLoader(train_data_blichen, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_blichen = torch.utils.data.DataLoader(val_data_blichen,  num_workers = 0, batch_size=batch_size)


# define model
blichen_model = models.resnet18(pretrained=False)
blichen_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'blichen_model.pt')
torch.save(blichen_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_blichen)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



blichen_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    blichen_model = torch.load(PATH)
    
    #prep to train
    blichen_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_blichen):  
        
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
        outputs = blichen_model(input_img)
        
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
    train_loss = train_loss/len(tl_blichen.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    blichen_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_blichen):
                        
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
            outputs = blichen_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(blichen_model, PATH)


traintext.close()
valtext.close()



#####################################################################################################################


#algae on rocks, abbr alg


train_alg = os.path.join(tf_dir, 'alga.txt')
val_alg = os.path.join(vf_dir, 'alga.txt')

traintext = open(train_alg, "r")
valtext = open(val_alg, "r")


# define datasets:

train_data_alg = AntarcticPlotDataset(traintext, train_dir, transform=train_transform)
val_data_alg = AntarcticPlotDataset(valtext, val_dir, transform=test_transform)


#test_data = datasets.ImageFolder("./output_data_rocks", transform=test_transform)

# load the data in batches: 

tl_alg = torch.utils.data.DataLoader(train_data_alg, num_workers = 0, batch_size=batch_size, shuffle=False)

vl_alg = torch.utils.data.DataLoader(val_data_alg,  num_workers = 0, batch_size=batch_size)


# define model
alg_model = models.resnet18(pretrained=False)
alg_model.fc = nn.Linear(in_features=512, out_features=1)

# save model
PATH = os.path.join(model_dir, 'alg_model.pt')
torch.save(alg_model, PATH)


# number of epochs to train the model, number of iterations per epoch
n_iterations = int(len(train_data_alg)/batch_size)

# lists to keep track of training progress:
train_loss_progress = []
validation_accuracy_progress = []



alg_model.train() # final step to prep model for training

for epoch in range(n_epochs):
    
    # monitor training loss
    train_loss = 0.0
    
    #load the model
    alg_model = torch.load(PATH)
    
    #prep to train
    alg_model.train()
    
    
    ###################
    # train the model #
    ###################
    
    for iter, D in enumerate(tl_alg):  
        
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
        outputs = alg_model(input_img)
        
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
    train_loss = train_loss/len(tl_alg.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    
    
    ##################
    #VALIDATION STAGE#
    
    # define variables
    correct = 0
    total = 0
    
    #prep for evaluation
    alg_model.eval() 
    
    
    
    with torch.no_grad(): #not exactly sure what this does
        for iter, D in enumerate(vl_alg):
                        
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
            outputs = alg_model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            # does the addition
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

    torch.save(alg_model, PATH)


traintext.close()
valtext.close()




#################################################################################################################
#### THE END 
