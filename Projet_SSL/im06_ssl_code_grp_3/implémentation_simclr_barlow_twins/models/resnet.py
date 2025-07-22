import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

def create_encoder(pretrained=True):
    if pretrained:
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    return encoder