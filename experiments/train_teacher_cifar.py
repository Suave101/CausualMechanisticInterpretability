# experiments/train_teacher_cifar.py
"""
Trains a CIFAR teacher with a standard recipe (SGD + Nesterov momentum, cosine LR, crop/flip
augmentation) and saves it where the experiment suite expects it.

Uses the 40k training split from get_dataset, so the 10k validation split used for causal profiling
stays unseen. Example:
    python experiments/train_teacher_cifar.py --model resnet50 --epochs 200 --seed 0
"""
import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.DataLoader import get_dataset
from src.models.resnet_cifar import ResNet18CIFAR, ResNet50CIFAR


def default_checkpoint(model_name, dataset, seed):
    suffix = f"_seed{seed}" if seed else ""
    return os.path.join("checkpoints", f"{model_name}_{dataset}{suffix}.pth")


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="output path (default: checkpoints/<model>_<dataset>[_seedN].pth)")
    parser.add_argument("--max_batches", type=int, default=None, help="limit batches per epoch (smoke tests only)")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, _, test_loader, num_classes, _, _ = get_dataset(args.dataset, batch_size=args.batch_size)
    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    test_loader = DataLoader(test_loader.dataset, batch_size=256, shuffle=False, **loader_kwargs)

    model_cls = ResNet50CIFAR if args.model == "resnet50" else ResNet18CIFAR
    model = model_cls(num_classes=num_classes, load_pretrained=False).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    steps_per_epoch = args.max_batches or len(train_loader)
    # Linear warmup over the first 5% of steps, then cosine decay to zero
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
    )
    criterion = nn.CrossEntropyLoss()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    ckpt_path = args.checkpoint or default_checkpoint(args.model, args.dataset, args.seed)
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    print(f"Training {args.model}/{args.dataset} seed {args.seed} for {args.epochs} epochs -> {ckpt_path} (bf16={use_bf16})", flush=True)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * y.size(0); seen += y.size(0)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            acc = evaluate(model, test_loader, device, max_batches=args.max_batches)
            print(f"  epoch {epoch+1:4d}/{args.epochs} | loss {running/seen:.4f} | test acc {acc:.2f}% | {time.time()-t0:.0f}s", flush=True)
            torch.save(model.state_dict(), ckpt_path)

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved teacher to {ckpt_path} (final test acc {evaluate(model, test_loader, device, max_batches=args.max_batches):.2f}%)", flush=True)


if __name__ == "__main__":
    main()
