# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np

from skimage.io import imread
from skimage.transform import resize

import torch
from torch.utils.data import Dataset, DataLoader

class KUSDataset(Dataset):
    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super(KUSDataset, self).__init__()
        self.root = root
        
        self.image_path = os.path.join(root, 'train_img')
        self.mask_path = os.path.join(root, 'train_mask')
        
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        
        
        self.len = 2323
        
    def __getitem__(self, index):
        images = os.listdir(self.image_path)
        masks = os.listdir(self.mask_path)
       
        img_data = imread(os.path.join(self.image_path, images[index]), as_gray=True)
        img_data = resize(img_data,(210, 290))
        img_data = img_data.astype('float32')
                
        mask_data = imread(os.path.join(self.mask_path, images[index]), as_gray=True)
        mask_data = resize(mask_data,(210, 290))
        mask_data = mask_data.astype('float32')
             
        
        if self.transforms is not None:
            img_data, mask_data = self.transforms(img_data, mask_data)
        else:
            seed = np.random.randint(2147483647)
        
        if self.transform is not None:
            random.seed(seed)
            img_data = self.transform(img_data)
        
        if self.target_transform is not None:
            random.seed(seed)
            mask_data = self.target_transform(mask_data)
             
        return img_data, mask_data
        
    def __len__(self):
        return self.len
    

    
    