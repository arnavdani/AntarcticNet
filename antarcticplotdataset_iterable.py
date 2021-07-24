import pandas as pd
import torchvision.io 
from torch.utils.data import TensorDataset, DataLoader, Dataset, IterableDataset
import os
from skimage import io
import cv2
import numpy as np 
import math
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms

# reference - Dataset is the superclass that is being extended and overridden
class AntarcticPlotDataset(Dataset):
    
    #initializing the master array
    newdata = []
    
    def __init__(self, txt_file, root_dir, transform=None):
        
        
        #directory where all the images are
        self.root_dir = root_dir
        
        #transforms to do on the images
        self.transform = transform
        
        #init
        self.newdata = []
        
        
        #counter for next() method, no longer used
        self.counter = -1
        
        #text file w/ target data for the images
        photoData = txt_file
        
        
        
        #reading the text file line by line 
        
        for line in photoData:
            info = line
            
    
            infolist = info.split(" ")
            
            #data from one line
            finaldata = []
            
            #since all the sub images for one image rest in a master folder, need to get inside that folder
            imgname = infolist[0]
            
            #seperating the .png from the image name
            rawImg = imgname.split("-")[0]
            
            #going into the folder named after the prent image
            img_dir = os.path.join(root_dir, rawImg)
            
            #using the text file to get the name of the image
            img = io.imread(os.path.join(img_dir, imgname))
            
            
            #formatting and transforming the images
            img = np.array(img)
            trans = transforms.ToPILImage()
            img = trans(img)
            img = self.transform(img)
            finaldata.append(img)
            
            
            
            #extracting the target data and adding it to the array for the line
            finaldata.append(int(info.split(":")[1].strip('\n')))
            
            
            #adding the information from one line to the full master array
            self.newdata.append(finaldata)


            
            
                    
    def __len__(self):
        return len(self.newdata)
    
    
    def __getitem__(self, i):
        
         #gets the ith item in newdata
        
         #error check
         if (i < 0 or i > len(self.newdata)):
            print("problem")
            return None
        
         else:
            
            #extracting image      
            img = self.newdata[i][0]
            
            #extracting target
            landmarks = int(self.newdata[i][1])
            
            #storing both image in target in a dictionary format
            sample = {'image' : img, 'landmarks' : landmarks}
            
            #returning the dictionary - the enumerator wants a dictionary object
            return sample
        
    
    
    
    
                              ######################################################
                              ######################################################
    ############################ EVERYTHING BELOW THIS IS OLD STUFF/GARBAGE ################################
                              ######################################################
                              ######################################################
    
    
    #FOR ITERABLE DATASET, NOT USED 
    def __iter__(self):
        
        print("iter was called")
        worker_info = torch.utils.data.get_worker_info()
        self.start = 5
        self.end = 4
        if worker_info is None:
            
            iter_start = self.start
            iter_end = self.end
        else:
            
            per_worker = int(math.ceil((self.end - self.start)/ float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        
        return iter(range(iter_start, iter_end))
        
    def __next__(self):
        self.counter = self.counter + 1
        return getitem(counter)
        
        

    
    
