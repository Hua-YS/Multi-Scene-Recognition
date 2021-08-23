#import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import *

class KFB(nn.Module):
    def __init__(self, model, num_classes, k=20):
        super(KFB, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.backbone = VGGNet(model, False).vggnet
        self.conv6 = nn.Conv2d(512, k*num_classes, 3, padding=1)
        self.conv7 = nn.Conv2d(k*num_classes, num_classes, 3, padding=1)
        self.conv8 = nn.Conv2d(k*num_classes, k*num_classes, 2)
        # pooling operations
        self.gmp = nn.MaxPool2d(14, 14) # global max pooling
        self.aap = nn.AdaptiveAvgPool2d((1, 1)) # adaptive average pooling
        self.mp = nn.MaxPool2d(2, 1) # 2d max pooling
        self.ccp = nn.AvgPool1d(k, k) # cross-channel pooling
        # norm, relu, fc
        self.norm = nn.BatchNorm2d(k*num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # vggnet
        c4 = self.backbone[:23](x)
        c5 = self.backbone[:30](x)
        
        # pred 1
        pred1 = self.fc(self.gmp(c5).view(-1, 512))

        # pred 2
        heatmap = self.gmp(self.conv6(c4))
        pred2 = self.aap(self.conv7(heatmap)).view(-1, self.num_classes)

        # pred 3
        x = self.relu(self.norm(self.conv8(heatmap)))
        pred3 = self.ccp(x.view(-1, 1, self.k*self.num_classes)).squeeze(1)

        return pred1+pred2+pred3

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.conv6.parameters(), 'lr': lr},
                {'params': self.conv7.parameters(), 'lr': lr},
                {'params': self.conv8.parameters(), 'lr': lr},
                {'params': self.fc.parameters(), 'lr': lr}]


