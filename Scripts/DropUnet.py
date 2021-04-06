# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def create_upconv(in_channels, out_channels, size=None, kernel_size=None, stride=None, padding=None):
    return nn.Sequential(
        nn.Upsample(size=size, mode='nearest')
        , nn.Conv2d(in_channels,out_channels,3,1,1)
        , nn.ReLU(inplace=True)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.ReLU(inplace=True)
        )

class DropUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_l1 = nn.Sequential(
            nn.Conv2d(1,32,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(32,32,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(64,64,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(128,128,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(256,256,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(512,512,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u4 = create_upconv(in_channels=512, out_channels=256, size=(26,36))

        self.conv_u4 = nn.Sequential(
            nn.Conv2d(512,256,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(256,256,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u3 = create_upconv(in_channels=256, out_channels=128, size=(52,72))

        self.conv_u3 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(128,128,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u2 = create_upconv(in_channels=128, out_channels=64, size=(105,145))

        self.conv_u2 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(64,64,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.deconv_u1 = create_upconv(in_channels=64, out_channels=32, size=(210,290))

        self.conv_u1 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1)
            , nn.ReLU(inplace=True)
            , nn.Conv2d(32,32,3,1,1)
            , nn.ReLU(inplace=True)
            )

        self.conv1x1_out = nn.Conv2d(32, 1, 1, 1, 0, bias=True)
        self.smout = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        input7 = self.dropout(input7)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        input8 = self.dropout(input8)

        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        input9 = self.dropout(input9)

        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        out = self.conv1x1_out(output9)
        
        return self.smout(out)