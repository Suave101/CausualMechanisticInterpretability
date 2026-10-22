# src/utils/CMI_Method.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def compute_causal_fingerprints_and_scores(model, val_loader, device):
    """Calculates activation fingerprints and single-unit causal ablation scores."""
    model.eval()
    val_imgs = torch.cat([x for x, _ in val_loader]).to(device)
    
    with torch.no_grad(), torch.amp.autocast("cuda"):
        baseline_logits = model(val_imgs)
        activations = model.get_layer_activations(val_imgs)

    fps = []
    unit_map = []  # Maps global_idx -> (layer_idx, local_unit_idx)
    
    for l_idx, act in enumerate(activations):
        # FC activation: [B, C]; Conv activation: [B, C, H, W] -> average spatially
        if act.dim() == 4:
            fp = act.mean(dim=(2, 3)).T.cpu().numpy()
        else:
            fp = act.T.cpu().numpy()
        fps.append(fp)
        for u_idx in range(act.shape[1]):
            unit_map.append((l_idx, u_idx))

    all_fps = np.vstack(fps)
    # Handle zero activation vectors for cosine stability
    zero_mask = np.all(all_fps == 0, axis=1)
    if np.any(zero_mask):
        all_fps[zero_mask] = 1e-8

    ablation_scores = np.zeros(len(unit_map))
    
    # Added tqdm progress bar to track individual unit ablation forward passes
    with torch.no_grad():
        for global_idx, (l_idx, u_idx) in enumerate(tqdm(unit_map, desc="Computing Causal Ablation Scores", unit="unit")):
            with torch.amp.autocast("cuda"):
                ablated_logits = model.forward_with_ablation(val_imgs, l_idx, u_idx)
                sim = F.cosine_similarity(baseline_logits, ablated_logits, dim=1).mean().item()
            ablation_scores[global_idx] = 1.0 - sim

    return all_fps, ablation_scores, unit_map

def rearrange_neurons_front_loaded(ablation_scores, top_n_pct=15.0):
    """
    Identifies the top n% most critical neurons across the network 
    and assigns them to Layer 1 (front layers).
    """
    total_units = len(ablation_scores)
    n_top = int(np.ceil((top_n_pct / 100.0) * total_units))
    
    sorted_indices = np.argsort(ablation_scores)[::-1]  # Highest causal score first
    top_n_indices = sorted_indices[:n_top]
    remaining_indices = sorted_indices[n_top:]
    
    return top_n_indices, remaining_indices
