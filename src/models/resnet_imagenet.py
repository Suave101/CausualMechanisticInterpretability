# src/models/resnet_imagenet.py
import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import RearrangeableModel

class ResNetImageNet(RearrangeableModel):
    def __init__(self, variant: str = "resnet18", pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        
        if variant == "resnet18":
            self.backbone = models.resnet18(weights=weights)
        elif variant == "resnet50":
            self.backbone = models.resnet50(weights=weights)
        elif variant == "resnet101":
            self.backbone = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

    def forward(self, x):
        return self.backbone(x)

    def get_layer_activations(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        l1 = self.backbone.layer1(x)
        l2 = self.backbone.layer2(l1)
        l3 = self.backbone.layer3(l2)
        l4 = self.backbone.layer4(l3)
        return [l1, l2, l3, l4]

    def forward_with_ablation(self, x, layer_idx, unit_idx):
        # Forward pass through backbone layers with specified unit ablated
        activations = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        layers = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        for idx, layer in enumerate(layers):
            x = layer(x)
            if idx == layer_idx:
                x = x.clone()
                x[:, unit_idx, :, :] = 0.0

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.fc(x)

    def get_total_units(self):
        # Total channels across residual stages
        if isinstance(self.backbone, models.ResNet18):
            return 64 + 128 + 256 + 512
        return 256 + 512 + 1024 + 2048  # Bottleneck architectures (50/101)