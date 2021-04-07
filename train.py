# -*- coding: utf-8 -*-

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms as tf

from Scripts import KUSDataloader, Loss, DropUnet

#####################################################################################################################################################
img_rows = 210
img_cols = 290

MAX_INTENSITY = 1.0
data_shuffle = True

train_data_path = "./Dataset"
#####################################################################################################################################################
transform_train = tf.Compose([
         tf.ToPILImage(),
         tf.RandomAffine(0, shear=[-15, 15, -15, 15]),
         tf.ToTensor()
     ])

transform_valid = tf.Compose([
         tf.ToPILImage(),
         tf.ToTensor()
     ])

dataset_train = KUSDataloader.KUSDataset(root=train_data_path, transforms=None, transform=transform_train, target_transform=transform_train)
dataset_val = KUSDataloader.KUSDataset(root=train_data_path, transforms=None, transform=transform_valid, target_transform=transform_valid)

n_total_sample=len(dataset_train)
n_train_sample = int(n_total_sample * 0.8)
n_val_sample=n_total_sample-n_train_sample
print("[Total:%d] [Train:%d] [Validate:%d]" % (n_total_sample, n_train_sample, n_val_sample))

train_valid_idx = np.load(os.path.join(train_data_path, 'train_valid_idx.npz'))
train_idx = train_valid_idx['train_idx']
valid_idx = train_valid_idx['valid_idx']

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
#####################################################################################################################################################
lr = 5e-4
n_epoch = 100
batch_size = 2

print("Dropout Unet Experiment]")

train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler, pin_memory=True)

device = torch.device("cuda:0")
criterion = Loss.DiceLoss()
#####################################################################################################################################################
for test_idx in range(1,6):
    print("="*40, end=" ")
    print("Test #%d" % test_idx, end=" ")
    print("="*40)
    
    model = DropUnet.DropUnet() 
    model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
        
    best_f1 = 0
    for epoch in range(n_epoch):
        running_loss=0
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
        running_loss /= len(train_loader)
    
        if (epoch+1) % 1 == 0:
            accuracy = 0
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            with torch.no_grad():
                model.eval()
                correct = 0
                for x, y in valid_loader:
                    x = x.float().to(device)
                    y = y >= 0.5
                    y = y.to(device)
            
                    output = model(x)
                    pred = output >= 0.5
                    correct += (y == pred).sum().item()
                
                    pred = pred.view(-1)
                
                    Trues = pred[pred == y.view(-1)]
                    Falses = pred[pred != y.view(-1)]
            
                    TP += (Trues == 1).sum().item()
                    TN += (Trues == 0).sum().item()
                    FP += (Falses == 1).sum().item()
                    FN += (Falses == 0).sum().item()
                
            accuracy = correct / (n_val_sample*442*565)
        
            if TP == 0:
                pass
            else:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 >= best_f1:
                    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss), end=" ")
                    print("[Accuracy:%f]" % accuracy, end=" ")
                    print("[Precision:%f]" % precision, end=" ")
                    print("[Recall:%f]" % recall, end=" ")
                    print("[F1 score:%f] **Best**" % f1)
                
                    best_f1 = f1
                    model_name = "./Models/" + str(test_idx) + ".pth"
                    torch.save({'model_state_dict': model.state_dict()}, model_name)

            model.train()