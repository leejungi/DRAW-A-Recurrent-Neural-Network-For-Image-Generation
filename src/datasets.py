import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(Dataset, self).__init__()
        self.data = data.data
        self.targets = data.targets
            
        self.data = self.data/255.0
        self.data = self.data.view(-1,28*28).float()
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.targets[index]
        return data, label 
