import sys
import numpy as np
import cv2
import os

class TextFileGenerator():
    

    dir_path = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/train_subsets'


    save_path = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/'



    img_data = open("C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/txtdata/trainingfile.txt", "w+")
    
    for filename in os.listdir(dir_path):
    
    
    
        set_folder = os.path.join(dir_path, filename)
        #print(filename)
        for img in os.listdir(set_folder):
            
            img_data.write(str(img) + " x:\n")






            


    

        