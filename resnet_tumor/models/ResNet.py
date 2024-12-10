import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class AvgPool(nn.Module):
    def forward(self, x):
        # print(x.shape)
        return F.avg_pool2d(x, x.shape[2:])#.squeeze()


class ResNet50(nn.Module):
    def __init__(self, num_outputs, fine_tuning=True):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
            nn.Dropout(0.5),
            layer4
        )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(2048, num_outputs)
        if fine_tuning:
            for param in self.resnet.parameters():
                param.requires_grad = False

            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.resnet.parameters():
                param.requires_grad = True

    def forward(self, x):
        out = self.resnet(x) # B, C
        return out

class ResNet50_seg_t2(nn.Module):
    def __init__(self, num_outputs, fine_tuning=True):
        super(ResNet50_seg_t2, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        weights = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # self.resnet.load_state_dict(weights.backbone.state_dict())
        for name, param in weights.state_dict().items():
            if 'backbone' not in name:
                continue
            resnet_name = name.replace('backbone.', '')
            self.resnet.state_dict()[resnet_name].copy_(param)
            # print(f'set params in {resnet_name} to {name}')
        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
            nn.Dropout(0.5),
            layer4
        )
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(2048, num_outputs)
        if fine_tuning:
            for param in self.resnet.parameters():
                param.requires_grad = False

            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

            for param in self.resnet.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.resnet.parameters():
                param.requires_grad = True

    def forward(self, x):
        out = self.resnet(x) # B, C
        return out