import torch.nn as nn
import torchvision.models as models

class VGGNet(nn.Module):
    def __init__(self, model, is_baseline=False):
        super(VGGNet, self).__init__()
        self.is_baseline = is_baseline
        self.model = model
        self.vggnet = model.features
        self.avg = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = model.classifier[:-1]

    def forward(self, x):
        if self.is_baseline:
            # using fc1, fc2
            return self.classifier(self.avg(self.vggnet(x)).view(-1, 25088))
        else:
            return self.vggnet(x)


class InceptionV3(nn.Module):
    def __init__(self, model):
        super(InceptionV3, self).__init__()
        self.model = model
        self.inceptionv3 = nn.Sequential(
            self.model.Conv2d_1a_3x3,
            self.model.Conv2d_2a_3x3,
            self.model.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.model.Conv2d_3b_1x1,
            self.model.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.model.Mixed_5b,
            self.model.Mixed_5c,
            self.model.Mixed_5d,
            self.model.Mixed_6a,
            self.model.Mixed_6b,
            self.model.Mixed_6c,
            self.model.Mixed_6d,
            self.model.Mixed_6e,
            self.model.Mixed_7a,
            self.model.Mixed_7b,
            self.model.Mixed_7c
            )

    def forward(self, x):
        return self.inceptionv3(x)


class ResNet(nn.Module):
    def __init__(self, model):    
        super(ResNet, self).__init__()
        self.model = model
        self.resnet = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
            )
                
    def forward(self, x):
        return self.resnet(x)

class SqueezeNet(nn.Module):
    def __init__(self, model):
        super(SqueezeNet, self).__init__()
        self.squeezenet = model.features

    def forward(self, x):
        return self.squeezenet(x)

class DenseNet(nn.Module):
    def __init__(self, model):
        super(DenseNet, self).__init__()
        self.model = model
        self.densenet = model.features

    def forward(self, x):
        return self.densenet(x)


class ShuffleNetV2(nn.Module):
    def __init__(self, model):
        super(ShuffleNetV2, self).__init__()
        self.model = model
        self.shufflenetv2 = nn.Sequential(
            self.model.conv1,
            self.model.maxpool,
            self.model.stage2,
            self.model.stage3,
            self.model.stage4,
            self.model.conv5
            )

    def forward(self, x):
        return self.shufflenetv2(x)

class MobileNetV2(nn.Module):
    def __init__(self, model):
        super(MobileNetV2, self).__init__()
        self.model = model
        self.mobilenetv2 = model.features

    def forward(self, x):
        return self.mobilenetv2(x)

class ResNeXt(nn.Module):
    def __init__(self, model):    
        super(ResNeXt, self).__init__()
        self.model = model
        self.resnext = nn.Sequential(
            self.model.conv1,                
            self.model.bn1,
            self.model.relu,                
            self.model.maxpool,                
            self.model.layer1,                
            self.model.layer2,                
            self.model.layer3,                
            self.model.layer4                
            )

    def forward(self, x):        
        return self.resnext(x)


class WideResNet(nn.Module):
    def __init__(self, model):
        super(WideResNet, self).__init__()
        self.model = model
        self.wideresnet = nn.Sequential(
            self.model.conv1,
            self.model.bn1,                
            self.model.relu,    
            self.model.maxpool,    
            self.model.layer1,    
            self.model.layer2,    
            self.model.layer3,    
            self.model.layer4
            )
        
    def forward(self, x):
        return self.wideresnet(x)


class MNASNet(nn.Module):
    def __init__(self, model):
        super(MNASNet, self).__init__()
        self.model = model
        self.mnasnet = model.layers

    def forward(self, x):
        return self.mnasnet(x)


class AtrousResNet(nn.Module):
    def __init__(self, model):
        super(AtrousResNet, self).__init__()
        self.model = model
        # modified to dilation version
        self.model.layer4[0].conv2.stride = (1, 1)
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[1].conv1.dilation = (2, 2)
        self.model.layer4[1].conv2.padding = (2, 2)
        self.model.layer4[1].conv2.dilation = (2, 2)
        self.model.layer4[1].conv3.dilation = (2, 2)
        self.model.layer4[2].conv1.dilation = (2, 2)
        self.model.layer4[2].conv2.padding = (2, 2)
        self.model.layer4[2].conv2.dilation = (2, 2)
        self.model.layer4[2].conv3.dilation = (2, 2)
        # *****************************
        self.atrous_resnet = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
            )
        
    def forward(self, x):
        return self.atrous_resnet(x)

class AtrousVGGNet(nn.Module):
    def __init__(self, model):
        super(AtrousVGGNet, self).__init__()
        self.atrous_vggnet = nn.Sequential(
            model.features[:-1]
            )

    def forward(self, x):
        return self.atrous_vggnet(x)

