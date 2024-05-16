'''
Author: Hadlay Zhang
Date: 2024-05-12 17:56:03
LastEditors: Hadlay Zhang
LastEditTime: 2024-05-16 13:40:03
FilePath: /root/MedicalVQA-RAD/image.py
Description: Image Encoder for extracting visual features
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.models.vision_transformer import vit_l_16, ViT_L_16_Weights
from torchvision.models.swin_transformer import Swin_V2_B_Weights
import numpy as np

# Here three models are used as examples: ConvNeXt, ViT-L-16, SwinT-V2-B
# For other pretrained models, please check: https://pytorch.org/vision/0.18/models.html

def get_Image_Encoder(image):
    if image == 'ConvNeXt':
        model = ConvNeXtEncoder()
    elif image == 'ViTL16':
        model = ViTL16Encoder()
    elif image == 'SwinTV2B':
        model = SwinTV2BEncoder()
    else:
        raise ValueError("Unknown Image Encoder")
    return model

class ConvNeXtEncoder(nn.Module):
    def __init__(self, pretrained=True, weights=ConvNeXt_Large_Weights.IMAGENET1K_V1):
        super(ConvNeXtEncoder, self).__init__()
        self.convnext_pretrained = convnext_large(pretrained=pretrained, weights=weights)
        # freeze weights
        for param in self.convnext_pretrained.parameters():
            param.requires_grad = False
        # remove pooling, softmax layers
        self.convnext_feature_extractor = nn.Sequential(*list(self.convnext_pretrained.children())[:-2])

    def forward(self, v):
        outputs = self.convnext_feature_extractor(v)
        outputs = torch.flatten(outputs, start_dim=1)
        outputs = outputs.unsqueeze(1)
        return outputs

class ViTL16Encoder(nn.Module):
    def __init__(self, pretrained=True, weights=ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1):
        super(ViTL16Encoder, self).__init__()
        self.vit_pretrained = models.vit_l_16(weights=weights)
        # freeze weights
        for param in self.vit_pretrained.parameters():
            param.requires_grad = False
        # remove pooling, softmax layers
        self.vit_feature_extractor = nn.Sequential(*list(self.vit_pretrained.children())[:-2])

    def forward(self, v):
        outputs = self.vit_feature_extractor(v)
        outputs = torch.flatten(outputs, start_dim=1)
        outputs = outputs.unsqueeze(1)
        return outputs

class SwinTV2BEncoder(nn.Module):
    def __init__(self, pretrained=True, weights=Swin_V2_B_Weights.IMAGENET1K_V1):
        super(SwinTV2BEncoder, self).__init__()
        if pretrained:
            self.swin_pretrained = models.swin_v2_b(weights=weights)
        else:
            self.swin_pretrained = models.swin_v2_b(weights=None)
        # freeze weights
        for param in self.swin_pretrained.parameters():
            param.requires_grad = False

        self.swin_feature_extractor = nn.Sequential(*list(self.swin_pretrained.children())[:-2])

    def forward(self, v):
        outputs = self.swin_feature_extractor(v)
        outputs = torch.flatten(outputs, start_dim=1)
        outputs = outputs.unsqueeze(1)
        return outputs
