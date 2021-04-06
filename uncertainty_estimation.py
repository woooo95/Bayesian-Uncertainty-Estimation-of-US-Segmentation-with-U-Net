# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as tf
import numpy as np
import cv2 as cv
from PIL import Image

import glob
import os
import pickle
import random

from skimage.io import imread
from skimage.transform import resize

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
print(valid_idx)
#####################################################################################################################################################
test_idx = "1"

device = torch.device("cuda:0")

model = DropUnet.DropUnet()
model.to(device)

model_name = "./Models/" + test_idx + ".pth"


checkpoint = torch.load(model_name)

from collections import OrderedDict
import matplotlib.pyplot as plt

new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
#####################################################################################################################################################
n_samples = 30
uncertainty_threshold = 0.1

# model.eval()
model.train()

binarise_weight = torch.ones(size=(1,1,2,2)).cuda()
binarise_bias = torch.zeros(1).cuda()        

with torch.no_grad():

    for i in valid_idx:
        print(i, end="  ")        
        x, y = dataset_val.__getitem__(i)
        x, y = x.float().to(device), y.float().to(device)
        x1, y1 = x.unsqueeze(0), y.unsqueeze(0)

        #### parameter sampling ###
        temp = []
        for _ in range(n_samples):    
            output = model(x1)
            temp.append(output)
        pred_samples = torch.cat(temp, dim=0)

        ### Prediction & Uncertainty map ###
        pred_mean = torch.mean(pred_samples, dim=0)
        prediction = pred_mean >= 0.5
        uncertainty_map = torch.var(pred_samples, dim=0)

        prediction = prediction.unsqueeze(0)
        uncertainty_map = uncertainty_map.float().unsqueeze(0)

        ### Uncertainty Metrics ###
        predictive_entropy = -(pred_mean * torch.log(pred_mean)) -(1-(pred_mean) * torch.log(1-(pred_mean)))
        sum_1 = pred_samples * torch.log(pred_samples)
        sum_1 = torch.sum(sum_1, dim=0)
        sum_2 = (1 - pred_samples) * torch.log(1 - pred_samples)
        sum_2 = torch.sum(sum_2, dim=0)
        mutual_information = predictive_entropy + (1/n_samples)*(sum_1 + sum_2)

        ### Performance Evaluation Metrics ###
        binarise_prediction = prediction == y1
        binarise_prediction = binarise_prediction.float()
        binarise_prediction = torch.nn.functional.conv2d(binarise_prediction, weight=binarise_weight, bias=binarise_bias, stride=(2,2)) / 4
        binarise_prediction = binarise_prediction > 0.5  # 1=Accurate, 0=Inaccurate
        
        binarise_uncertaintiy_map = torch.nn.functional.conv2d(uncertainty_map, weight=binarise_weight, bias=binarise_bias, stride=(2,2)) / 4
        binarise_uncertaintiy_map = binarise_uncertaintiy_map > uncertainty_threshold  # 1=Uncertain, 0=Certain
        
        n_ac = torch.sum(binarise_prediction & ~binarise_uncertaintiy_map)
        n_au = torch.sum(binarise_prediction & binarise_uncertaintiy_map)
        n_ic = torch.sum(~binarise_prediction & ~binarise_uncertaintiy_map)
        n_iu = torch.sum(~binarise_prediction & binarise_uncertaintiy_map)
        n_total = n_ac + n_au + n_ic + n_iu

        p_ac = n_ac / (n_ac + n_ic)
        p_ui = n_iu / (n_ic + n_iu)
        pavpu = (n_ac + n_iu) / n_total
        
        pr_a = (n_ac + n_iu) / n_total
        pr_e = (n_ac + n_au)/n_total * (n_ac + n_ic)/n_total + (n_iu + n_ic)/n_total * (n_iu + n_au)/n_total 
        kappa = (pr_a - pr_e) / (1 - pr_e)


        ### Custom Metrics ###
        threshold_uncertainty_map = uncertainty_map > uncertainty_threshold

        intersection = prediction & y1.type(torch.bool)
        union = prediction | y1.type(torch.bool)
        IoU = intersection.sum() / union.sum()

        certain_intersection = prediction & ~threshold_uncertainty_map & y1.type(torch.bool)
        certain_union = (prediction & ~threshold_uncertainty_map) | y1.type(torch.bool)
        certain_IoU = certain_intersection.sum() / certain_union.sum()

        uncertain_intersection = (prediction | threshold_uncertainty_map) & y1.type(torch.bool)
        uncertain_union = prediction | threshold_uncertainty_map | y1.type(torch.bool)
        uncertain_IoU = uncertain_intersection.sum() / uncertain_union.sum()

        # # only positive
        # TP = prediction & y1.type(torch.bool)        # True Positive
        # FP = prediction & ~y1.type(torch.bool)       # False Positive
        # CP = prediction & (uncertainty_map <= 0.4)   # Certain Positive
        # UP = prediction & (uncertainty_map > 0.4)    # Uncertain Positive

        # n_ac = torch.sum(TP & CP)
        # n_au = torch.sum(TP & UP)
        # n_ic = torch.sum(FP & CP)
        # n_iu = torch.sum(FP & UP)
        # n_total = torch.sum(prediction)

        # p_ac = n_ac / (n_ac + n_ic)
        # p_ui = n_iu / (n_ic + n_iu)
        # pavpu = (n_ac + n_iu) / n_total

        ### Save Result ###
        result = f"[P(accurate|certain)={p_ac:.4f}] [P(uncertain|inaccurate)={p_ui:.4f}] [PAVPU={pavpu:.4f}] [kappa={kappa:.4f}] [IoU={IoU:.4f}] [certain IoU={certain_IoU:.4f}] [uncertain IoU={uncertain_IoU:.4f}]"
        print(result)

        image_name = f"Inference/val({i}).png"

        fig = plt.figure(figsize=(14,8))
        plt.suptitle(f"val({i} - {result}")
        plt.subplot(231)
        plt.title(f"({i}) Input Image")
        plt.imshow(x1.squeeze().detach().cpu().numpy(), cmap="gray")
        plt.subplot(232)
        plt.title(f"({i}) Label")
        plt.imshow(y1.squeeze().detach().cpu().numpy(), cmap="gray")
        plt.subplot(233)
        plt.title(f"({i}) Prediction")
        plt.imshow(prediction.squeeze().detach().cpu().numpy(), cmap="gray")
        plt.subplot(234)
        plt.title(f"({i}) Uncertainty Map")
        plt.imshow(uncertainty_map.squeeze().detach().cpu().numpy(), cmap="gray")
        plt.subplot(235)
        plt.title(f"({i}) Predictive Entropy")
        plt.imshow(predictive_entropy.squeeze().detach().cpu().numpy(), cmap="gray")
        plt.subplot(236)
        plt.title(f"({i}) Mutual Information")
        plt.imshow(mutual_information.squeeze().detach().cpu().numpy(), cmap="gray")
        plt.savefig(image_name)
        plt.close(fig)
        # plt.show()
        # break
