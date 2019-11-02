
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:48:25 2019

@author: kk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['Inception_AttentionNet', 'UWNet']


path = './Inet.pkl'

def UWNet(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:

        model = Inception_AttentionNet(**kwargs)
        model.load_state_dict(torch.load(path))
        return model

    return Inception_AttentionNet(**kwargs)




class Inception_AttentionNet(nn.Module):
    
    def __init__(self,num_classes=2):
        super(Inception_AttentionNet, self).__init__()
        self.features = nn.Sequential(
                BasicConv2d(3,16,kernel_size=7,stride=2),#147*147*16
                BasicConv2d(16,32,kernel_size=3,stride=1,padding=1),#147*147*32
                nn.MaxPool2d(kernel_size=3, stride=2),#73*73*32
                Inception_1(32,32),#36*36*128
                Inception_2(128,128),#17*17*512
                Inception_3(512,256),#8*8*1024
                )
        
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(1024 * 8 * 8, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1024, num_classes),
                )
        
    def forward(self,x):       
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x






class attention_1(nn.Module):
    def __init__(self,channels,size1):
        super(attention_1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((8,8))
        self.conv2d1 = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        self.bn = nn.BatchNorm2d(channels,eps=0.001, momentum=0.1,affine=True)

        self.conv2d2 = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

        self.upsamping = nn.UpsamplingBilinear2d(size=size1)
        
        
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.conv2d1(x)
        x = self.bn(x)
        x = F.relu(x, inplace = True)
        x = self.conv2d2(x)
        x = self.bn(x)
        x = torch.sigmoid(x)
        x = self.upsamping(x)
        
        return module_input*x + module_input 
    #    input,in_channels=128,size=(36,36)
    
class attention_2(nn.Module):
    def __init__(self,channels,size1):
        super(attention_2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((8,8))
        self.conv2d1 = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        self.bn = nn.BatchNorm2d(channels,eps=0.001, momentum=0.1,affine=True)

        self.conv2d2 = nn.Conv2d(channels,channels,kernel_size=1,padding=0)

        self.upsamping = nn.UpsamplingBilinear2d(size=size1)
        
        
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.conv2d1(x)
        x = self.bn(x)
        x = F.relu(x, inplace = True)
        x = self.conv2d2(x)
        x = self.bn(x)
        x = torch.sigmoid(x)
        x = self.upsamping(x)
        
        return module_input*x + module_input 



class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception_1(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(Inception_1, self).__init__()
        self.branch1x1_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch1x1_2 = BasicConv2d(8, 32, kernel_size=3 , stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(8, 32, kernel_size=5, padding=2)
        self.branch5x5_3 = BasicConv2d(32, 32, kernel_size=3,  stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(8, 32, kernel_size=7, padding=3)
        self.branch3x3dbl_3 = BasicConv2d(32, 32, kernel_size=3, stride=2)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)
        branch1x1= self.branch1x1_2(branch1x1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)



class Inception_2(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception_2, self).__init__()
        self.branch1x1_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch1x1_2 = BasicConv2d(64, 128, kernel_size=3 , stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(64, 128, kernel_size=5, padding=2)
        self.branch5x5_3 = BasicConv2d(128, 128, kernel_size=3,  stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 128, kernel_size=7, padding=3)
        self.branch3x3dbl_3 = BasicConv2d(128, 128, kernel_size=3, stride=2)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)
        branch1x1= self.branch1x1_2(branch1x1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)



class Inception_3(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception_3, self).__init__()
        self.branch1x1_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch1x1_2 = BasicConv2d(128, 256, kernel_size=3 , stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(128, 256, kernel_size=5, padding=2)
        self.branch5x5_3 = BasicConv2d(256, 256, kernel_size=3,  stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(128, 256, kernel_size=7, padding=3)
        self.branch3x3dbl_3 = BasicConv2d(256, 256, kernel_size=3, stride=2)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)
        branch1x1= self.branch1x1_2(branch1x1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

        
        


