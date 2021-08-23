#import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import *

class VGGNetBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(VGGNetBaseline, self).__init__()
        self.backbone = VGGNet(model, True)
        self.fc = nn.Linear(4096, num_classes, bias=True)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # vggnet
        x = self.backbone(x)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]


class InceptionBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(InceptionBaseline, self).__init__()
        self.backbone = nn.Sequential(
            InceptionV3(model),
            nn.AdaptiveAvgPool2d((1, 1))
            )
        self.fc = nn.Linear(model.fc.in_features, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # inceptionv3
        x = torch.flatten(self.backbone(x), 1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]


class ResNetBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(ResNetBaseline, self).__init__()
        self.backbone = nn.Sequential(
            ResNet(model),
            nn.AdaptiveAvgPool2d((1, 1))
            )
        self.fc = nn.Linear(model.fc.in_features, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # resnet
        x = torch.flatten(self.backbone(x), 1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]


class SqueezeNetBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(SqueezeNetBaseline, self).__init__()
        self.backbone = nn.Sequential(
            SqueezeNet(model),
            nn.Dropout(p=0.5, inplace=False)
            )
        self.fc_conv = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # squeezenet
        x = self.backbone(x)
        return torch.flatten(self.avg(self.fc_conv(x)), 1)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc_conv.parameters(), 'lr': lr}]


class DenseNetBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(DenseNetBaseline, self).__init__()
        self.backbone = nn.Sequential(
            DenseNet(model),
            nn.AdaptiveAvgPool2d((1, 1))                                                      
            )
        self.fc = nn.Linear(model.classifier.in_features, num_classes)

        # image normalization        
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # densenet
        x = torch.flatten(self.backbone(x), 1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]



class ShuffleNetBaseline(nn.Module):        
    def __init__(self, model, num_classes):
        super(ShuffleNetBaseline, self).__init__()        
        self.backbone = nn.Sequential(        
            ShuffleNetV2(model),            
            nn.AdaptiveAvgPool2d((1, 1))            
            )            
        self.fc = nn.Linear(model.fc.in_features, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # shufflenet
        x = torch.flatten(self.backbone(x), 1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]


class MobileNetBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(MobileNetBaseline, self).__init__()
        self.backbone = nn.Sequential(
            MobileNetV2(model),
            nn.AdaptiveAvgPool2d((1, 1))
            )
        self.fc = nn.Linear(model.classifier[1].in_features, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # mobilenet
        x = torch.flatten(self.backbone(x), 1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]



class ResNeXtBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(ResNeXtBaseline, self).__init__()
        self.backbone = nn.Sequential(
            ResNeXt(model),            
            nn.AdaptiveAvgPool2d((1, 1))                        
            )
        self.fc = nn.Linear(model.fc.in_features, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # resnext
        x = torch.flatten(self.backbone(x), 1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]

class MNASNetBaseline(nn.Module):
    def __init__(self, model, num_classes):
        super(MNASNetBaseline, self).__init__()
        self.backbone = nn.Sequential(
            MNASNet(model),
            nn.AdaptiveAvgPool2d((1, 1))    
            )
        self.fc = nn.Linear(model.classifier[1].in_features, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        # mnasnet
        x = torch.flatten(self.backbone(x), 1)
        return self.fc(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.backbone.parameters(), 'lr': lr * lrp},
                {'params': self.fc.parameters(), 'lr': lr}]

