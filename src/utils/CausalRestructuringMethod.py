# src/utils/CausalRestructuringMethod.py
import copy
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.utils.CMI_Method import compute_causal_fingerprints_and_scores


def make_safe_fps(fps):
    """Ensures activation fingerprints are non-zero float64 vectors normalized to unit length."""
    fps_safe = fps.copy().astype(np.float64)
    norms = np.linalg.norm(fps_safe, axis=1, keepdims=True)
    zero_mask = (norms < 1e-12).squeeze()
    
    if np.ndim(zero_mask) == 0:
        zero_mask = np.array([zero_mask])

    if np.any(zero_mask):
        fps_safe[zero_mask] += np.random.normal(0, 1e-7, size=fps_safe[zero_mask].shape)
        norms = np.linalg.norm(fps_safe, axis=1, keepdims=True)

    return fps_safe / (norms + 1e-12)


def find_causal_medoid(fps, ablation_scores, members, causal_weight=0.5):
    """
    Selects cluster medoid balancing fingerprint distance and causal score.
    causal_weight (alpha) in [0.0, 1.0]:
      - 0.0 = Pure geometric centroid distance
      - 1.0 = Pure max causal score selection
    """
    if len(members) == 1:
        return members[0]
        
    sub_fps = fps[members]
    centroid = sub_fps.mean(axis=0)
    
    dists = np.array([
        1.0 - np.dot(fp, centroid) / (np.linalg.norm(fp) * np.linalg.norm(centroid) + 1e-8)
        for fp in sub_fps
    ])
    
    sub_scores = ablation_scores[members]
    max_score = np.max(sub_scores) + 1e-8
    penalty = 1.0 - (sub_scores / max_score)
    
    combined_score = (1.0 - causal_weight) * dists + causal_weight * penalty
    
    return members[np.argmin(combined_score)]


def cluster_background_units(
    fingerprints, 
    ablation_scores, 
    unit_map, 
    reserved_units, 
    reduction_rate=0.5, 
    causal_weight=0.5
):
    kept_units = set(reserved_units)
    layer_cluster_map = {}
    
    layer_to_units = {}
    for global_idx, (l_idx, u_idx) in enumerate(unit_map):
        if l_idx not in layer_to_units:
            layer_to_units[l_idx] = []
        layer_to_units[l_idx].append(global_idx)
        
    for l_idx, layer_global_indices in layer_to_units.items():
        layer_cluster_map[l_idx] = []
        
        for g_idx in layer_global_indices:
            if g_idx in reserved_units:
                u_idx = unit_map[g_idx][1]
                layer_cluster_map[l_idx].append((u_idx, [u_idx]))

        bg_units = [g_idx for g_idx in layer_global_indices if g_idx not in reserved_units]
        if len(bg_units) == 0:
            continue
            
        target_k = max(1, int(round(len(bg_units) * (1.0 - reduction_rate))))
        
        if target_k >= len(bg_units):
            for g_idx in bg_units:
                kept_units.add(g_idx)
                u_idx = unit_map[g_idx][1]
                layer_cluster_map[l_idx].append((u_idx, [u_idx]))
        else:
            sub_fps = fingerprints[bg_units]
            fps_safe = make_safe_fps(sub_fps)
            
            clusterer = AgglomerativeClustering(n_clusters=target_k, metric="euclidean", linkage="average")
            labels = clusterer.fit_predict(fps_safe)
            sub_scores = ablation_scores[bg_units]
            
            for c_id in range(target_k):
                members_sub_idx = np.where(labels == c_id)[0]
                if len(members_sub_idx) == 0:
                    continue
                
                medoid_sub_idx = find_causal_medoid(
                    fps_safe, sub_scores, members_sub_idx, causal_weight=causal_weight
                )
                medoid_global = bg_units[medoid_sub_idx]
                kept_units.add(medoid_global)
                
                medoid_local = unit_map[medoid_global][1]
                members_local = [unit_map[bg_units[idx]][1] for idx in members_sub_idx]
                
                layer_cluster_map[l_idx].append((medoid_local, members_local))
                
    return sorted(list(kept_units)), layer_cluster_map


def extract_target_layers(model: nn.Module):
    """
    Extracts primary parametric layers (nn.Conv2d, nn.Linear) in execution order,
    skipping shortcut/downsample side branches to maintain 1:1 alignment with activation indices.
    """
    if hasattr(model, "get_target_layers") and callable(model.get_target_layers):
        return model.get_target_layers()
    if hasattr(model, "target_layers"):
        return model.target_layers

    target_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if "downsample" not in name and "shortcut" not in name:
                target_layers.append(m)
    return target_layers


def apply_step4_weight_merging(model: nn.Module, layer_cluster_map: dict):
    """
    Executes Step 4 Weight Merging:
    1. Incoming Weight Averaging
    2. Outgoing Weight Folding scaled by sqrt(|m|)
    3. Zeroing out non-representative weights
    """
    param_layers = extract_target_layers(model)
    
    with torch.no_grad():
        for l_idx, clusters in layer_cluster_map.items():
            if l_idx >= len(param_layers):
                continue
                
            curr_layer = param_layers[l_idx]
            has_next = (l_idx + 1) < len(param_layers)
            next_layer = param_layers[l_idx + 1] if has_next else None
            
            can_fold_outgoing = False
            if next_layer is not None:
                curr_out_channels = curr_layer.weight.shape[0]
                next_in_channels = next_layer.weight.shape[1] if next_layer.weight.dim() >= 2 else 0
                if curr_out_channels == next_in_channels:
                    can_fold_outgoing = True
            
            for medoid_u, members_u in clusters:
                if len(members_u) <= 1:
                    continue
                    
                m_size = len(members_u)
                sqrt_m = np.sqrt(m_size)
                non_medoid_u = [u for u in members_u if u != medoid_u]
                
                # Bounds check against layer output dimension
                if medoid_u >= curr_layer.weight.shape[0] or any(u >= curr_layer.weight.shape[0] for u in non_medoid_u):
                    continue

                # 1. Incoming Weight Averaging
                curr_layer.weight.data[medoid_u] = curr_layer.weight.data[members_u].mean(dim=0)
                if curr_layer.bias is not None:
                    curr_layer.bias.data[medoid_u] = curr_layer.bias.data[members_u].mean(dim=0)
                
                # 2. Outgoing Weight Folding into Medoid Column
                if can_fold_outgoing:
                    if max(members_u) < next_layer.weight.shape[1]:
                        if isinstance(next_layer, nn.Linear):
                            folded_outgoing = next_layer.weight.data[:, non_medoid_u].sum(dim=1) / sqrt_m
                            next_layer.weight.data[:, medoid_u] += folded_outgoing
                            next_layer.weight.data[:, non_medoid_u] = 0.0
                        elif isinstance(next_layer, nn.Conv2d):
                            folded_outgoing = next_layer.weight.data[:, non_medoid_u, :, :].sum(dim=1) / sqrt_m
                            next_layer.weight.data[:, medoid_u, :, :] += folded_outgoing
                            next_layer.weight.data[:, non_medoid_u, :, :] = 0.0

                # 3. Zero out non-medoid incoming parameters
                curr_layer.weight.data[non_medoid_u] = 0.0
                if curr_layer.bias is not None:
                    curr_layer.bias.data[non_medoid_u] = 0.0


def recalibrate_batchnorm(model, val_loader, mask_vector, device, num_batches=15):
    """Recalibrates BatchNorm statistics post-restructuring."""
    model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            x = x.to(device)
            if hasattr(model, "forward_with_mask"):
                _ = model.forward_with_mask(x, mask_vector)
            else:
                _ = model(x)
    model.eval()


def evaluate_masked_model(
    sub_model, 
    teacher_model, 
    test_loader, 
    kept_units, 
    total_units, 
    device, 
    val_loader=None
):
    """Evaluates zero-shot test accuracy and teacher agreement."""
    mask_vector = torch.zeros(total_units, device=device)
    mask_vector[list(kept_units)] = 1.0

    if val_loader is not None:
        recalibrate_batchnorm(sub_model, val_loader, mask_vector, device)
    
    sub_model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    correct_zs = 0
    correct_agr = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if teacher_model is not None:
                teacher_logits = teacher_model(x)
                teacher_preds = teacher_logits.argmax(dim=1)
            else:
                teacher_preds = y
            
            if hasattr(sub_model, "forward_with_mask"):
                zs_logits = sub_model.forward_with_mask(x, mask_vector)
            else:
                zs_logits = sub_model(x)
                
            zs_preds = zs_logits.argmax(dim=1)
            
            correct_zs += (zs_preds == y).sum().item()
            correct_agr += (zs_preds == teacher_preds).sum().item()
            total_samples += y.size(0)
            
    zs_acc = (correct_zs / total_samples) * 100.0
    zs_agr = (correct_agr / total_samples) * 100.0
    
    print("\n--- [Causal Restructuring Zero-Shot Evaluation] ---")
    print(f"  -> Units Retained: {len(kept_units)} / {total_units} ({len(kept_units)/total_units*100:.1f}%)")
    print(f"  -> Zero-Shot Accuracy: {zs_acc:.2f}%")
    print(f"  -> Zero-Shot Teacher Agreement: {zs_agr:.2f}%")
    
    return {"zs_acc": zs_acc, "zs_agr": zs_agr}


def perform_causal_restructuring(
    teacher_model,
    val_loader=None,
    test_loader=None,
    top_n_pct=15.0,
    reduction_rate=0.5,
    causal_weight=0.5,
    device=torch.device("cpu"),
    cache_dir="cache",
    model_id="mlp3_uci_digits",
    all_fps=None,
    ablation_scores=None,
    unit_map=None,
    **kwargs
):
    if test_loader is None and val_loader is not None:
        test_loader = val_loader
    if val_loader is None and test_loader is not None:
        val_loader = test_loader

    if all_fps is not None and ablation_scores is not None and unit_map is not None:
        fingerprints = all_fps
    else:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{model_id}_causal_cache.pt")

        if os.path.exists(cache_path):
            print(f"--> [Cache Hit] Loading cached ablation scores & fingerprints from {cache_path}")
            cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
            ablation_scores = cache_data["ablation_scores"]
            fingerprints = cache_data["fingerprints"]
            unit_map = cache_data["unit_map"]
        else:
            print(f"--> [Cache Miss] Computing activation fingerprints and causal ablation scores...")
            fingerprints, ablation_scores, unit_map = compute_causal_fingerprints_and_scores(
                teacher_model, val_loader, device, max_profile_samples=256
            )

            torch.save({
                "ablation_scores": ablation_scores,
                "fingerprints": fingerprints,
                "unit_map": unit_map
            }, cache_path)

    if isinstance(ablation_scores, torch.Tensor):
        ablation_scores = ablation_scores.cpu().numpy()
    if isinstance(fingerprints, torch.Tensor):
        fingerprints = fingerprints.cpu().numpy()

    total_units = len(ablation_scores)
    k_reserve = max(1, int(round(total_units * (top_n_pct / 100.0))))
    
    sorted_unit_indices = np.argsort(ablation_scores)[::-1]
    reserved_units = set(sorted_unit_indices[:k_reserve])
    
    kept_units, layer_cluster_map = cluster_background_units(
        fingerprints=fingerprints,
        ablation_scores=ablation_scores,
        unit_map=unit_map,
        reserved_units=reserved_units,
        reduction_rate=reduction_rate,
        causal_weight=causal_weight
    )

    mask_vector = torch.zeros(total_units, device=device)
    mask_vector[kept_units] = 1.0

    sub_model = copy.deepcopy(teacher_model).to(device)
    apply_step4_weight_merging(sub_model, layer_cluster_map)

    zs_metrics = evaluate_masked_model(
        sub_model=sub_model,
        teacher_model=teacher_model,
        test_loader=test_loader,
        kept_units=kept_units,
        total_units=total_units,
        device=device,
        val_loader=val_loader
    )

    return sub_model, kept_units, mask_vector, zs_metrics
