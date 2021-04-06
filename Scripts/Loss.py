# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        
        smooth = 0
           
        num = target.shape[0]
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        intersection=torch.mul(m1,m2)
        score = 1-torch.sum((2. * torch.sum(intersection,dim=1) + smooth) / (torch.sum(m1,dim=1) + torch.sum(m2,dim=1) + smooth))/num
        
        return score