class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, **kwargs):
        super().__init__()


        self.loss = loss
        self.backbone = tvmodels.__dict__[name](pretrained=pretrained)
        if 'vit' in name:
            # For ViT models, the classifier attribute is different
            self.feature_dim = self.backbone.heads[0].in_features
            self.backbone.heads = nn.Identity()
        else:
            self.feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
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


def vit_b_16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vit_b_16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def densenet161(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "densenet161",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def googlenet(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "googlenet",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html
