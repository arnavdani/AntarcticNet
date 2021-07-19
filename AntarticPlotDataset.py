import pandas as pd




class AntarticPlotDataset(Dataset):
    
    
    def __init__(self, csv_file, root_dir, transform=None)
    
        self.photoData = open(csv_file)
        reader = csv.reader(csv_file)
        
        
        self.root_dir = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/output_data_set4'
        
        self.transform = transform
        
    
    def __len__(self):
        return len(self.photoData)
    
    
    def __getitem__(self, i)
    
    