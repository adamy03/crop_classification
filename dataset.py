import numpy as np
import pandas as pd
import h5py
import os
import torch
from torch.utils.data import Dataset

class CropDataset(Dataset):
    """Standard dataset for pytorch dataloaders. 
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data_path, transform=None):
        self.path = data_path
        h5f = h5py.File(self.path, 'r')
        self.len = len(h5f[list(h5f.keys())[0]])
        h5f.close()
        self.transform = transform
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        sample = h5f['samples'][index]
        labels = h5f['labels'][index]
        labels = one_hot_encode(labels)
        h5f.close()
        sample, labels = torch.Tensor(sample.astype(float)).type(torch.float), torch.Tensor(labels.astype(int)).type(torch.int)
        
        if self.transform:
            # Ensure labels undergo same transformation 
            fused = torch.cat((sample, labels))
            fused = self.transform(fused)
            sample, labels = fused[0:5], fused[5:8].reshape(3, fused.shape[1], fused.shape[2])
        
        return sample, labels
    
def one_hot_encode(labels):
    """One hot encode data: convert classes in 2D to 3D with each channel representing positive for a corresponding class.
    """
    out = np.zeros((3, labels.shape[1], labels.shape[2]), dtype=int)
    for i in range(labels.shape[1]):
        for j in range(labels.shape[2]):
            out[int(labels[0, i, j]), i, j] = 1
    return out

def un_hot_encode(encoded):
    """Convert one hot encoded outputs 2D class label image. Class value corresponds to channel.
    """
    out = np.zeros((1, encoded.shape[1], encoded.shape[2]), dtype=int)
    for c in range(encoded.shape[0]):
        for i in range(encoded.shape[1]):
            for j in range(encoded.shape[2]):
                if encoded[c, i, j]:
                    out[0, i, j] = c 
    return out

def divide_image(image, tile_size, stride):
    """Splits images into tiles of specified size with given stride.
    """
    tiles = []
    height, width, = image.shape[1], image.shape[2]
    tile_height, tile_width = tile_size
    stride_vertical, stride_horizontal = stride

    for y in range(0, height - tile_height + 1, stride_vertical):
        for x in range(0, width - tile_width + 1, stride_horizontal):
            tile = image[::, y:y+tile_height, x:x+tile_width]
            tiles.append(tile)

    return tiles