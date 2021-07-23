import pandas as pd
import torchvision.io 
from torch.utils.data import TensorDataset, DataLoader, Dataset, IterableDataset
import os
from skimage import io
import cv2
import numpy as np  


class AntarcticPlotDataset(IterableDataset):
    
    
    newdata = []
    
    def __init__(self, txt_file, root_dir, transform=None):
        
        
        self.root_dir = root_dir
        self.transform = transform
        self.newdata = []
        
        photoData = txt_file
        for line in photoData:
            info = line
            infolist = info.split(",")
            finaldata = []
            imgname = infolist[0] + ".jpg"
            #print(os.path.join(root_dir, imgname))
            img = io.imread(os.path.join(root_dir, imgname))
            img = np.array(img)
                       
            finaldata.append(img)
            
            for i in range(9):
                
                finaldata.append(infolist[i+1].split("-")[1].strip('\n'))
            
            self.newdata.append(finaldata)


            
            
                    
    def __len__(self):
        return len(self.newdata)
    
    
    def __getitem__(self, i):
        
         #cut off all except 0th index
         if (i < 0 or i > len(self.newdata)):
            print("problem")
            return None
        
         else:       
            img = self.newdata[i][0]
            #landmarks = self.newdata.iloc[i, 1:]
            self.newdata[i].pop(0)
            landmarks = self.newdata[i]
            sample = {'image': img, 'landmarks': landmarks}
            return sample
        
    
    
        

    
    
