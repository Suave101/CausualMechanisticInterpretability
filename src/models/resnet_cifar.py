# src/models/resnet_cifar.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import RearrangeableModel

class ResNet18CIFAR(RearrangeableModel):
    def __init__(self, num_classes=10, load_pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256, num_classes)

        if load_pretrained and num_classes == 10:
            print("[Pretrained] Loading CIFAR-10 pre-trained weights from PyTorch Hub...")
            hub_model = torch.hub.load("huyvnphan/PyTorch_CIFAR10", "cifar10_resnet18", pretrained=True)
            # Adapt weights to our architecture layout
            self._copy_hub_weights(hub_model)

    def _copy_hub_weights(self, hub_model):
        with torch.no_grad():
            self.conv1.weight.copy_(hub_model.conv1.weight)
            self.bn1.weight.copy_(hub_model.bn1.weight)
            self.bn1.bias.copy_(hub_model.bn1.bias)
            self.conv2.weight.copy_(hub_model.layer1[0].conv1.weight)
            self.bn2.weight.copy_(hub_model.layer1[0].bn1.weight)
            self.bn2.bias.copy_(hub_model.layer1[0].bn1.bias)

    def forward(self, x):
        h1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2, 2)
        h2 = F.max_pool2d(F.relu(self.bn2(self.conv2(h1))), 2, 2)
        h3 = F.adaptive_avg_pool2d(F.relu(self.bn3(self.conv3(h2))), (1, 1))
        return self.fc(torch.flatten(h3, 1))

    def get_layer_activations(self, x):
        h1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2, 2)
        h2 = F.max_pool2d(F.relu(self.bn2(self.conv2(h1))), 2, 2)
        h3 = F.adaptive_avg_pool2d(F.relu(self.bn3(self.conv3(h2))), (1, 1))
        return [h1, h2, h3]

    def forward_with_ablation(self, x, layer_idx, unit_idx):
        h1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2, 2)
        if layer_idx == 0:
            h1 = h1.clone()
            h1[:, unit_idx, :, :] = 0.0

        h2 = F.max_pool2d(F.relu(self.bn2(self.conv2(h1))), 2, 2)
        if layer_idx == 1:
            h2 = h2.clone()
            h2[:, unit_idx, :, :] = 0.0

        h3 = F.adaptive_avg_pool2d(F.relu(self.bn3(self.conv3(h2))), (1, 1))
        if layer_idx == 2:
            h3 = h3.clone()
            h3[:, unit_idx, :, :] = 0.0

        return self.fc(torch.flatten(h3, 1))

    def get_total_units(self):
        return 64 + 128 + 256