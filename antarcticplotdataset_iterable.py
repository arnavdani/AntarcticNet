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


class AntarcticPlotDataset(Dataset):
    
    
    newdata = []
    
    def __init__(self, txt_file, root_dir, transform=None):
        
        
        self.root_dir = root_dir
        self.transform = transform
        self.newdata = []
        self.counter = -1
        
        photoData = txt_file
        for line in photoData:
            info = line
            infolist = info.split(" ")
            finaldata = []
            imgname = infolist[0]
            rawImg = imgname.split("-")[0]
            #print(os.path.join(root_dir, imgname))
            img_dir = os.path.join(root_dir, rawImg)
            img = io.imread(os.path.join(img_dir, imgname))
            img = np.array(img)
            
            trans = transforms.ToPILImage()
            img = trans(img)
            img = self.transform(img)
            finaldata.append(img)
            finaldata.append(int(info.split(":")[1].strip('\n')))
            
            self.newdata.append(finaldata)


            
            
                    
    def __len__(self):
        return len(self.newdata)
    
    
    def __getitem__(self, i):
        
         #print("get item was called on index " + str(i))
        
         #cut off all except 0th index
         if (i < 0 or i > len(self.newdata)):
            print("problem")
            return None
        
         else:       
            img = self.newdata[i][0]
            
            #landmarks = self.newdata.iloc[i, 1:]
            self.newdata[i].pop(0)
            landmarks = self.newdata[i]
            target = [landmarks]
            target = torch.tensor(target)
            #sample = {'image': img, 'landmarks': landmarks}
            sample = {'image' : img, 'landmarks' : landmarks}
            
            return sample
        
    
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
        
        

    
    
