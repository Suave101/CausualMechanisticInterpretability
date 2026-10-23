# src/models/resnet_cifar.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import RearrangeableModel

# Verified public mirrors hosting pretrained CIFAR-10 ResNet-18 state dicts
WEIGHT_URLS = [
    "https://ndownloader.figshare.com/files/26676110",
    "https://huggingface.co/buffaX/CIFAR10_ResNet18/resolve/main/pytorch_model.bin"
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18CIFAR(RearrangeableModel):
    def __init__(self, num_classes=10, load_pretrained=True):
        super().__init__()
        self.in_planes = 64

        # CIFAR-10 ResNet-18 Stem (3x3 conv, stride 1, padding 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

        if load_pretrained:
            self._load_pretrained_weights()

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self):
        state_dict = None
        for url in WEIGHT_URLS:
            try:
                print(f"--> Attempting download of CIFAR-10 ResNet-18 weights from:\n    {url}")
                state_dict = torch.hub.load_state_dict_from_url(
                    url, 
                    map_location="cpu", 
                    progress=True
                )
                print("--> Download successful!")
                break
            except Exception as e:
                print(f"[Warning] Failed loading from {url} ({e})")

        if state_dict is not None:
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"--> Pretrained weights loaded. (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
        else:
            print("[Warning] Could not fetch pretrained weights from mirrors. Initializing model randomly.")

    def load_state_dict(self, state_dict, strict=True):
        """Dynamically normalizes state dict keys across different model wrappers and checkpoints."""
        cleaned = {}
        for k, v in state_dict.items():
            # Remove wrapper prefixes
            nk = k.replace("backbone.", "").replace("model.", "").replace("net.", "").replace("module.", "")
            # Map downsample keys to shortcut keys
            nk = nk.replace("downsample.", "shortcut.")
            # Map fc keys to linear
            if nk.startswith("fc."):
                nk = nk.replace("fc.", "linear.", 1)
            cleaned[nk] = v

        return super().load_state_dict(cleaned, strict=strict)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_layer_activations(self, x):
        activations = []
        out = F.relu(self.bn1(self.conv1(x)))
        activations.append(out)
        
        for layer_group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer_group:
                residual = out
                out = F.relu(block.bn1(block.conv1(out)))
                activations.append(out)
                out = block.bn2(block.conv2(out))
                activations.append(out)
                out = F.relu(out + block.shortcut(residual))
                
        return activations

    def forward_with_ablation(self, x, layer_idx, unit_idx):
        curr_layer = 0

        def ablate_if_match(tensor):
            nonlocal curr_layer
            if curr_layer == layer_idx:
                tensor = tensor.clone()
                if tensor.dim() == 4:
                    tensor[:, unit_idx, :, :] = 0.0
                else:
                    tensor[:, unit_idx] = 0.0
            curr_layer += 1
            return tensor

        out = F.relu(self.bn1(self.conv1(x)))
        out = ablate_if_match(out)

        for layer_group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer_group:
                residual = out

                out = F.relu(block.bn1(block.conv1(out)))
                out = ablate_if_match(out)

                out = block.bn2(block.conv2(out))
                out = ablate_if_match(out)

                out = F.relu(out + block.shortcut(residual))

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_with_mask(self, x, mask_vector):
        offset = 0

        def apply_layer_mask(tensor):
            nonlocal offset
            c_dim = tensor.shape[1]
            layer_mask = mask_vector[offset : offset + c_dim]
            offset += c_dim
            if tensor.dim() == 4:
                return tensor * layer_mask.view(1, c_dim, 1, 1)
            return tensor * layer_mask.unsqueeze(0)

        out = F.relu(self.bn1(self.conv1(x)))
        out = apply_layer_mask(out)

        for layer_group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer_group:
                residual = out

                out = F.relu(block.bn1(block.conv1(out)))
                out = apply_layer_mask(out)

                out = block.bn2(block.conv2(out))
                out = apply_layer_mask(out)

                out = F.relu(out + block.shortcut(residual))

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_total_units(self):
        # 64 + (64+64)*2 + (128+128)*2 + (256+256)*2 + (512+512)*2 = 3904
        return 64 + (64 + 64) * 2 + (128 + 128) * 2 + (256 + 256) * 2 + (512 + 512) * 2


def build_resnet18(num_classes=10, pretrained=True):
    return ResNet18CIFAR(num_classes=num_classes, load_pretrained=pretrained)
