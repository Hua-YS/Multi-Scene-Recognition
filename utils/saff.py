#import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import *

class SAFF(nn.Module):
    def __init__(self, model, num_classes, dim=512):
        super(SAFF, self).__init__()
        self.dim = dim
        self.backbone = VGGNet(model, False).vggnet
        self.fc1 = nn.Linear(1280, dim, bias=True)
        self.fc2 = nn.Linear(dim, num_classes, bias=True)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.norm = nn.BatchNorm1d(1280)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # vggnet
        eps = 1e-20
        c3 = self.pool1(self.backbone[:17](x))
        c4 = self.pool2(self.backbone[:24](x))
        c5 = self.backbone(x)
        x = torch.cat([c3, c4, c5], dim=1)
        b, c, h, w = x.shape
        
        
        # spatial weighted sum pooling
        s = torch.sum(x, 1)
        s = torch.sqrt(s/(s.sqrt().sum((1, 2), keepdim=True)+eps)).unsqueeze(1) # a: 0.5, b: 2
        x_s = torch.sum(s*x, (2, 3))

        # cross-dimensional weighting
        omega = torch.sum(x>0, (2, 3))/(w*h)
        wk = torch.log((c*eps+omega.sum(1, keepdim=True))/(eps+omega)+eps)
        x = x_s*wk

        ''' 
        # pca whitening (fail training) 
        x = x.t()
        x = x-torch.mean(x, dim=0, keepdim=True)
        cov = torch.mm(x, x.t())/c
        u, s, v = torch.svd(cov)
        x = torch.diag(2./(torch.sqrt(s[:self.dim])+eps)).mm(u[:, :self.dim].t()).mm(x).t()
        '''
        
        x = self.relu(self.fc1(self.norm(x)))
        return self.fc2(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc1.parameters(), 'lr': lr},
                {'params': self.fc2.parameters(), 'lr': lr}]


