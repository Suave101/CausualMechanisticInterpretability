import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import RearrangeableModel

class ResNet18CIFAR(RearrangeableModel):
    def __init__(self, num_classes=10, load_pretrained=True):
        super().__init__()
        # 1. Build standard ResNet18 adapted for 32x32 CIFAR-10 image resolution
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)

        # 2. Direct download of pre-trained CIFAR-10 weights bypassing hubconf zip extraction
        if load_pretrained and num_classes == 10:
            try:
                print("[Pretrained] Direct loading CIFAR-10 ResNet18 weights from release URL...")
                weight_url = "https://github.com/huyvnphan/PyTorch_CIFAR10/releases/download/v0.0.2/cifar10_resnet18.pt"
                state_dict = torch.hub.load_state_dict_from_url(
                    weight_url, map_location="cpu", progress=True
                )
                self.backbone.load_state_dict(state_dict)
                print("[Pretrained] Successfully loaded pre-trained CIFAR-10 weights!")
            except Exception as e:
                print(f"[Pretrained Warning] Failed to fetch online weights ({e}). Will fall back to local training cache.")

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
        # Total convolutional channels across residual stages (64 + 128 + 256 + 512 = 960)
        return 64 + 128 + 256 + 512