import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .orconv import ORConv2d
from .ripool import RotationInvariantPooling


class ConvBlock(nn.Module):
    
    def __init__(self,input_channel,output_channel):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self,inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self,num_channel=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.feat_channels=num_channel
        self.or_conv = ORConv2d(
            self.feat_channels,
            int(self.feat_channels),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_pool = RotationInvariantPooling(256, 8)
    def forward(self,inp):
        feat=self.layers(inp)
        or_feat=self.or_conv(feat)
        or_feat=self.or_pool(or_feat)
        
        return or_feat
if __name__ == '__main__':

    model = BackBone()
    data = torch.randn(2, 3, 84, 84)
    x = model(data)
    #print(x.size())
    #print(x.shape)