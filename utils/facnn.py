#import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import *

class FACNN(nn.Module):
    def __init__(self, model, num_classes, dim=2048):
        super(FACNN, self).__init__()
        self.dim = dim
        self.backbone = VGGNet(model, True).vggnet
        self.classifier = VGGNet(model, True).classifier
        self.fc = nn.Linear(4096+dim, num_classes, bias=True)
        self.conv = nn.Conv2d(1280, dim, 1)
        self.pool1 = nn.AvgPool2d(4, 4)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # vggnet
        # feature aggregation
        eps = 1e-8
        c3 = self.pool1(self.backbone[:17](x))
        c3 = c3/((c3*c3+eps).sum(1, keepdim=True).sqrt()+eps)
        c4 = self.pool2(self.backbone[:24](x))
        c4 = c4/((c4*c4+eps).sum(1, keepdim=True).sqrt()+eps)
        c5 = self.backbone(x)
        c5 = c5/((c5*c5+eps).sum(1, keepdim=True).sqrt()+eps)
        x_fa = torch.cat([c3, c4, c5], dim=1)
        x_fa = self.pool(self.relu(self.conv(x_fa))).view(-1, self.dim)

        # vgg16 fc2 output:
        x_fc2 = self.classifier(self.backbone(x).view(-1, 25088))

        # fusion
        x = torch.cat([x_fa, x_fc2], dim=1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters(), 'lr': lr},
                {'params': self.fc.parameters(), 'lr': lr},
                {'params': self.conv.parameters(), 'lr': lr}]


