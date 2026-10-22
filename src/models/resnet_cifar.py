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
    
    def forward_with_mask(self, x, mask_vector):
        """
        Applies a global 1D binary mask across all convolutional channels and FC hidden units.
        Channel masks are broadcasted spatially across [B, C, H, W] feature maps.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Track offset in the global mask vector
        offset = 0
        
        # Helper to apply mask to 4D tensor [B, C, H, W] or 2D tensor [B, C]
        def apply_layer_mask(tensor):
            nonlocal offset
            c_dim = tensor.shape[1]
            layer_mask = mask_vector[offset : offset + c_dim]
            offset += c_dim
            if tensor.dim() == 4:
                return tensor * layer_mask.view(1, c_dim, 1, 1)
            return tensor * layer_mask.unsqueeze(0)
    
        # Conv1 mask
        out = apply_layer_mask(out)
    
        # Process Layer Blocks
        for layer_group in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer_group:
                residual = out
                out = F.relu(block.bn1(block.conv1(out)))
                out = apply_layer_mask(out)
                out = block.bn2(block.conv2(out))
                out = apply_layer_mask(out)
                if block.shortcut is not None:
                    residual = block.shortcut(residual)
                out = F.relu(out + residual)
    
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
