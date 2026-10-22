# src/utils/checkpoint_utils.py
import os
import torch
import torch.nn as nn
import torch.optim as optim

CHECKPOINT_DIR = "./checkpoints"

def load_or_cache_model(
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    device: torch.device,
    train_loader=None,
    epochs: int = 5,
    force_train: bool = False
) -> nn.Module:
    """
    1. Checks if cached weights exist in `./checkpoints/`.
    2. If found (and force_train is False), loads cached weights directly.
    3. If missing, runs initial training pass (if train_loader provided) and saves weights.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_filename = f"{model_name}_{dataset_name}.pth"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_filename)

    # 1. Load from local cache if present
    if os.path.exists(ckpt_path) and not force_train:
        print(f"[Cache] Found cached weights at '{ckpt_path}'. Loading model...")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    # 2. If no cache found, generate weights via initial training
    if train_loader is not None:
        print(f"[Cache] No cached weights found at '{ckpt_path}'. Training baseline to populate cache...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.to(device)
        model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"  -> Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

        # Save to local cache
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Cache] Successfully saved weights to '{ckpt_path}'.")
        model.eval()
        return model

    raise FileNotFoundError(
        f"No cached checkpoint found at '{ckpt_path}' and no train_loader provided to generate one."
    )