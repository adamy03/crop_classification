import numpy as np
import pandas as pd
import h5py
import os
import torch
from torch.utils.data import Dataset

class CropDataset(Dataset):
    def __init__(self, data_path:str):
        self.path = data_path
        h5f = h5py.File(self.path, 'r')
        self.len = len(h5f[list(h5f.keys())[0]])
        h5f.close()
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        sample = h5f['samples'][index]
        labels = h5f['labels'][index]
        h5f.close()
        
        return torch.Tensor(sample.astype(float)), torch.Tensor(labels.astype(float))
        