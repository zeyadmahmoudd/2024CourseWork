# Copyright (c) EEEM071, University of Surrey

import torch.nn as nn
import torchvision.models as tvmodels
from torchvision.models import ViT_B_16_Weights

__all__ = ["mobilenet_v3_small", "vgg16", "vit_b_16"]

class TorchVisionModel(nn.Module):
    def __init__(self, model_func, weights, num_classes, loss, **kwargs):
        super().__init__()

        self.loss = loss

        # Load the model with specified weights
        if weights:
            self.backbone = model_func(weights=weights)
        else:
            self.backbone = model_func()

        # Adjusting the model depending on whether it's a Vision Transformer
        if isinstance(self.backbone, tvmodels.VisionTransformer):
            self.feature_dim = self.backbone.heads[1].in_features  # Usually, the classifier is the second last layer in Vision Transformers
            self.backbone.heads = nn.Identity()  # Replace the head with Identity
        else:
            # For CNNs like MobileNet and VGG
            if hasattr(self.backbone, 'classifier') and isinstance(self.backbone.classifier, nn.Sequential):
                self.feature_dim = self.backbone.classifier[0].in_features
                self.backbone.classifier = nn.Identity()  # Replace classifier with Identity
            elif hasattr(self.backbone, 'fc'):
                self.feature_dim = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        v = self.backbone(x)  # Process input through the backbone

        if not self.training:
            return v

        y = self.classifier(v)  # Apply classifier to the features

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")

def vgg16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vgg16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

def mobilenet_v3_small(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

def vit_b_16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vit_b_16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model 


# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html
