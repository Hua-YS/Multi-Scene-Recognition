import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from backbone import *

class RelationNet(nn.Module):
    def __init__(self, backbone, num_classes, num_moda, num_units):    
        super(RelationNet, self).__init__()
        self.num_classes = num_classes
        self.num_moda = num_moda
        self.num_units = num_units
        if backbone=='resnet50':
            self.backbone = AtrousResNet(models.resnet50(pretrained=True))
            self.backbone_feat = 2048
        elif backbone=='vggnet16':
            self.backbone = AtrousVGGNet(models.vgg16(pretrained=True))
            self.backbone_feat = 512

        self.feature_size = 14
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # relation module
        conv_parcels = []        
        stn_params = []
        g_theta = []
        for i in range(num_classes):
            conv_parcels.append(nn.Conv2d(self.backbone_feat, num_moda, kernel_size=1, stride=1, padding=0, bias=False))
            tmp = nn.Linear(int(num_moda*self.feature_size*self.feature_size/4), 6)
            nn.init.zeros_(tmp.weight)
            tmp.bias = torch.nn.Parameter(torch.Tensor([1, 0, 0, 0, 1, 0]))
            tmp.bias[1].require_grad=False 
            tmp.bias[3].require_grad=False
            stn_params.append(tmp)
            for j in range(num_classes-1):
                g_theta.append(nn.Conv2d(num_moda*2, num_units, kernel_size=1, stride=1, padding=0, bias=False))
            
        self.conv_parcels = nn.ModuleList(conv_parcels)
        self.stn_params = nn.ModuleList(stn_params)
        self.g_theta = nn.ModuleList(g_theta)
        self.f_phi = nn.Linear(num_units, 1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.backbone(x)

        # extract attentional regions of each parcel
        x_obj = []
        for i in range(self.num_classes):
            x_tmp = self.conv_parcels[i](x)
            tmp = F.max_pool2d(F.relu(x_tmp), 2).view(-1, int(self.num_moda*self.feature_size**2/4))
            tmp = self.stn_params[i](tmp).view(-1, 2, 3)
            affine_grid_points = F.affine_grid(tmp, torch.Size((tmp.size(0), self.num_moda, self.feature_size, self.feature_size)), align_corners=True)
            x_obj.append(F.grid_sample(x_tmp, affine_grid_points, align_corners=True))
        
        # build relations between each label pair
        idx_g = 0
        outputs = []
        for i in range(self.num_classes):
            relation_inter = 0
            # g_theta
            for j in range(self.num_classes):
                if not i == j:
                    relation_inter += F.relu(self.g_theta[idx_g](torch.cat([x_obj[i], x_obj[j]], axis=1)))
                    idx_g += 1
            relation_accum = self.avg(relation_inter).view(-1, self.num_units)
            # f_phi
            outputs.append(self.f_phi(relation_accum))

        outputs = torch.cat(outputs, axis=-1)
        
        return outputs

    def get_config_optim(self, lr, lrp):
        params = []       
        idx_g = 0
        for i in range(self.num_classes):
            params.append({'params': self.conv_parcels[i].parameters(), 'lr': lr})
            params.append({'params': self.stn_params[i].parameters(), 'lr': lr})
        for i in range(self.num_classes*(self.num_classes-1)):
            params.append({'params': self.g_theta[i].parameters(), 'lr': lr})

        params.append({'params': self.f_phi.parameters(), 'lr': lr})
        params.append({'params': self.backbone.parameters(), 'lr': lr * lrp})

        return params
        
'''
def rl_resnet50(num_classes, num_moda, num_units, pretrained=True):    
    model = models.resnet50(pretrained=pretrained)
    return RLResNet(model, num_classes, num_moda, num_units)

def rl_resnet101(num_classes, num_moda, num_units, pretrained=True):
    model = models.resnet101(pretrained=pretrained)
    return RLResNet(model, num_classes, num_moda, num_units)

def rl_resnet152(num_classes, num_moda, num_units, pretrained=True):
    model = models.resnet152(pretrained=pretrained)
    return RLResNet(model, num_classes, num_moda, num_units)
'''




