# Copyright (c) EEEM071, University of Surrey

import torch.nn as nn
import torchvision.models as tvmodels
from torchvision.models import VGG16_Weights, MobileNet_V3_Small_Weights, ViT_B_16_Weights

class TorchVisionModel(nn.Module):
    def __init__(self, model_func, num_classes, loss, weights=None, **kwargs):
        super().__init__()
        self.loss = loss
        self.backbone = model_func(weights=weights) if weights else model_func()

        # Handling for Vision Transformers
        if 'VisionTransformer' in model_func.__name__:
            # Attempt to access different possible names for the classifier head
            if hasattr(self.backbone, 'head'):
                self.feature_dim = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
            elif hasattr(self.backbone, 'fc'):  # Some models might use 'fc' instead of 'head'
                self.feature_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            else:
                raise AttributeError("Classifier head not found in Vision Transformer model")
        else:
            # Handling for traditional CNN models
            if hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential):
                self.feature_dim = self.backbone.classifier[0].in_features
                self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, 'fc'):
                self.feature_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        if not self.training:
            return features
        return self.classifier(features)

def vgg16(num_classes, loss={"xent"}, weights=VGG16_Weights.DEFAULT, **kwargs):
    return TorchVisionModel(tvmodels.vgg16, num_classes, loss, weights, **kwargs)

def mobilenet_v3_small(num_classes, loss={"xent"}, weights=MobileNet_V3_Small_Weights.DEFAULT, **kwargs):
    return TorchVisionModel(tvmodels.mobilenet_v3_small, num_classes, loss, weights, **kwargs)

def vit_b_16(num_classes, loss={"xent"}, weights=ViT_B_16_Weights.DEFAULT, **kwargs):
    return TorchVisionModel(tvmodels.vit_b_16, num_classes, loss, weights, **kwargs)

# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html
