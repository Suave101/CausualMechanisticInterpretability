# src/utils/DataLoader.py
"""
Dataset Image Transformations Summary:

1. UCI Digits ('uci_digits'):
   - Train / Val / Test:
     * Min-max scale pixel values to [0.0, 1.0] (division by 16.0).
     * Convert to 1D PyTorch FloatTensor (length 64).

2. CIFAR-10 & CIFAR-100 ('cifar10', 'cifar100'):
   - Train & Validation (via Subset split of full_train):
     * RandomCrop(32, padding=4)
     * RandomHorizontalFlip()
     * ToTensor() [converts PIL/ndarray to Tensor in range [0.0, 1.0]]
     * Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
   - Test:
     * ToTensor()
     * Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))

3. ImageNet-1K ('imagenet1k'):
   - Train / Val / Test (shared loader using validation split):
     * Resize(256)
     * CenterCrop(224)
     * ToTensor()
     * Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np

class DataTypes:
    UCI_DIGITS = "uci_digits"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    IMAGENET1K = "imagenet1k"

def resolve_cifar10_root(default_data_dir: str):
    """
    Auto-detects cluster dataset environment variables ($CIFAR10_ROOT or $AIP_DATASETS/cifar-10).
    Falls back to default_data_dir if environment variables are not found.
    """
    if "CIFAR10_ROOT" in os.environ and os.path.exists(os.environ["CIFAR10_ROOT"]):
        print(f"--> [Cluster Dataset] Using $CIFAR10_ROOT: {os.environ['CIFAR10_ROOT']}")
        return os.environ["CIFAR10_ROOT"], False
        
    if "AIP_DATASETS" in os.environ:
        aip_cifar_path = os.path.join(os.environ["AIP_DATASETS"], "cifar-10")
        if os.path.exists(aip_cifar_path):
            print(f"--> [Cluster Dataset] Using $AIP_DATASETS path: {aip_cifar_path}")
            return aip_cifar_path, False

    # Fallback to local default path
    return default_data_dir, True


def get_dataset(dataset: str, batch_size: int = 128, data_dir: str = "./data"):
    """
    Returns (train_loader, val_loader, test_loader, num_classes, input_channels, is_conv)
    """
    
    if dataset == DataTypes.UCI_DIGITS:
        digits = load_digits()
        X, y = digits.data / 16.0, digits.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            10, 64, False
        )
        
    elif dataset in [DataTypes.CIFAR10, DataTypes.CIFAR100]:
        is_c10 = (dataset == DataTypes.CIFAR10)
        ds_cls = torchvision.datasets.CIFAR10 if is_c10 else torchvision.datasets.CIFAR100
        num_classes = 10 if is_c10 else 100
        
        # Auto-resolve cluster paths for CIFAR-10
        if is_c10:
            effective_root, should_download = resolve_cifar10_root(data_dir)
        else:
            effective_root, should_download = data_dir, True

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        full_train = ds_cls(root=effective_root, train=True, download=should_download, transform=transform_train)
        test_set = ds_cls(root=effective_root, train=False, download=should_download, transform=transform_test)
        
        targets = np.array(full_train.targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
        
        return (
            DataLoader(Subset(full_train, train_idx), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(full_train, val_idx), batch_size=batch_size, shuffle=False),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
            num_classes, 3, True
        )

    elif dataset == DataTypes.IMAGENET1K:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_set = torchvision.datasets.ImageFolder(root=f"{data_dir}/imagenet/val", transform=transform)
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return loader, loader, loader, 1000, 3, True

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
