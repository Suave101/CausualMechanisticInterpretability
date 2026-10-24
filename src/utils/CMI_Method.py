# src/utils/CMI_Method.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def compute_causal_fingerprints_and_scores(
    model, 
    val_loader, 
    device, 
    max_profile_samples=256, 
    cache_dir="cache", 
    model_id=None
):
    """Calculates or loads cached activation fingerprints and single-unit causal ablation scores."""
    
    # 0. Check Disk Cache First
    if model_id is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{model_id}_causal_cache.pt")
        if os.path.exists(cache_path):
            print(f"--> [Cache Hit] Loading cached ablation scores & fingerprints from {cache_path}")
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
            return cache_data["fingerprints"], cache_data["ablation_scores"], cache_data["unit_map"]

    model.eval()
    
    # 1. Collect a fast calibration subset (default: 256 samples)
    val_imgs_list = []
    total_samples = 0
    for x, _ in val_loader:
        val_imgs_list.append(x)
        total_samples += x.size(0)
        if total_samples >= max_profile_samples:
            break
            
    val_imgs = torch.cat(val_imgs_list)[:max_profile_samples].to(device)

    # 2. Baseline Forward Pass & Feature Extraction
    with torch.no_grad(), torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
        baseline_logits = model(val_imgs)
        activations = model.get_layer_activations(val_imgs)

    fps = []
    unit_map = []  # Maps global_idx -> (layer_idx, local_unit_idx)
    
    for l_idx, act in enumerate(activations):
        if act.dim() == 4:
            fp = act.mean(dim=(2, 3)).T.cpu().numpy()
        else:
            fp = act.T.cpu().numpy()
        fps.append(fp)
        for u_idx in range(act.shape[1]):
            unit_map.append((l_idx, u_idx))

    all_fps = np.vstack(fps)
    zero_mask = np.all(all_fps == 0, axis=1)
    if np.any(zero_mask):
        all_fps[zero_mask] = 1e-8

    ablation_scores = np.zeros(len(unit_map))
    
    # 3. tqdm progress bar over unit ablations
    print(f"--> [Cache Miss] Computing activation fingerprints and causal ablation scores ({len(unit_map)} units)...")
    with torch.no_grad():
        for global_idx, (l_idx, u_idx) in enumerate(tqdm(unit_map, desc="Computing Causal Ablation Scores", unit="unit")):
            with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
                ablated_logits = model.forward_with_ablation(val_imgs, l_idx, u_idx)
                sim = F.cosine_similarity(baseline_logits, ablated_logits, dim=1).mean().item()
            ablation_scores[global_idx] = 1.0 - sim

    # 4. Save to Disk Cache
    if model_id is not None:
        torch.save({
            "ablation_scores": ablation_scores,
            "fingerprints": all_fps,
            "unit_map": unit_map
        }, cache_path)
        print(f"--> Saved causal metrics to {cache_path}")

    return all_fps, ablation_scores, unit_map

def rearrange_neurons_front_loaded(ablation_scores, top_n_pct=15.0):
    total_units = len(ablation_scores)
    n_top = int(np.ceil((top_n_pct / 100.0) * total_units))
    
    sorted_indices = np.argsort(ablation_scores)[::-1]
    top_n_indices = sorted_indices[:n_top]
    remaining_indices = sorted_indices[n_top:]
    
    return top_n_indices, remaining_indices
