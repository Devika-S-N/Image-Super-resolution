import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer='features_35', weight=1.0):
        super(VGGPerceptualLoss, self).__init__()
        self.weight = weight
        vgg = vgg19(pretrained=True).features
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg_layer = nn.Sequential(*list(vgg.children())[:36]).eval()
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        sr_features = self.vgg_layer(sr)
        hr_features = self.vgg_layer(hr)
        return self.weight * self.criterion(sr_features, hr_features)
