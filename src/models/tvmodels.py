# Copyright (c) EEEM071, University of Surrey

import torch.nn as nn
import torchvision.models as tvmodels


__all__ = ["mobilenet_v3_small", "vgg16"]


class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained=True, **kwargs):
        super().__init__()

        self.loss = loss

        # Dynamically load the model from torchvision's dictionary
        if pretrained:
            # Use pretrained weights if available
            model_func = tvmodels.__dict__[name]
            self.backbone = model_func(weights=model_func.DEFAULT)
        else:
            # Load the model without pretrained weights
            self.backbone = tvmodels.__dict__[name](pretrained=False)

        # Check if the model is a Vision Transformer
        if 'vit' in name:
            # For Vision Transformers, the final layer is usually named 'head'
            self.feature_dim = self.backbone.heads[0].in_features
            self.backbone.heads = nn.Identity()
        else:
            # For CNNs, replace the classifier
            self.feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()

        # New classifier for the specified number of classes
        self.classifier = nn.Linear(self.feature_dim, num_classes)
    def forward(self, x):
        v = self.backbone(x)

        if not self.training:
            return v

        y = self.classifier(v)

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


# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html
