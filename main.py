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


#circle crop data cleaning blah blah blah


rockpath = #define
soilpath = #define
MMpath = #define
LiPath = #define
BrySpoPath = #define
SanioniaPath = #define
grasspath = #define
PoStricPath = #define
ChorisoPath = #define
BrownLiPath = #define
algPath = #define


# load models

rock_model = torch.load(rockpath)
soil_model = torch.load(soilpath)
moribund_model = torch.load(MMpath)
wlichen_model = torch.load(LiPath)
bryum_model = torch.load(BrySpoPath)
sanionia_model = torch.load(SanioniaPath)
grass_model = torch.load(grasspath)
polytrich_model = torch.load(PoStricPath)
chor_model = torch.load(ChorisoPath)
blichen_model = torch.load(BrownLiPath)
alg_model = torch.load(algPath)




# load in image, ideally one subset of 
#image_location = #define
#img = cv2.imread(image_location)


#get outputs from models
weights = [rock_model(img), soil_model(img), moribund_model(img), wlichen_model(img), 
                    bryum_model(img), sanionia_model(img), grass_model(img), chor_model(img),
                    blichen_model(img), alg_model(img)]


legend = ["rock", "soil", "moribund moss", "white lichen", "bryum spo.", "sanionia", "hairgrass", 
                "polytrichium strictum", "chorisodontium aciphyllum", "brown lichen", "algae"]

def findType():
    max_index = weights.index(max(weights))
    return legend[max_index]

print(findType())