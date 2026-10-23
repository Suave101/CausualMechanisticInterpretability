# src/utils/CausalRestructuringMethod.py
import copy
import os
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.utils.CMI_Method import compute_causal_fingerprints_and_scores


def make_safe_fps(fps):
    """L2 unit-normalizes activation fingerprints and adds jitter to zero vectors."""
    fps_safe = fps.copy().astype(np.float64)
    norms = np.linalg.norm(fps_safe, axis=1, keepdims=True)
    zero_mask = (norms < 1e-12).squeeze()
    
    if np.ndim(zero_mask) == 0:
        zero_mask = np.array([zero_mask])

    if np.any(zero_mask):
        fps_safe[zero_mask] += np.random.normal(0, 1e-7, size=fps_safe[zero_mask].shape)
        norms = np.linalg.norm(fps_safe, axis=1, keepdims=True)

    return fps_safe / (norms + 1e-12)


def find_causal_medoid(fps, ablation_scores, members):
    """Selects the cluster representative that minimizes distance to centroid and maximizes causal score."""
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
    penalty = 0.5 * (1.0 - (sub_scores / max_score))
    
    return members[np.argmin(dists + penalty)]


def cluster_background_units(fingerprints, ablation_scores, unit_map, reserved_units, reduction_rate=0.5):
    """
    Partitions non-reserved background units layer-by-layer and selects 
    medoid units via agglomerative clustering.
    """
    kept_units = set(reserved_units)
    
    # Organize units by layer index
    layer_to_units = {}
    for global_idx, (l_idx, u_idx) in enumerate(unit_map):
        if l_idx not in layer_to_units:
            layer_to_units[l_idx] = []
        layer_to_units[l_idx].append(global_idx)
        
    for l_idx, layer_global_indices in layer_to_units.items():
        bg_units = [g_idx for g_idx in layer_global_indices if g_idx not in reserved_units]
        
        if len(bg_units) == 0:
            continue
            
        target_k = max(1, int(round(len(bg_units) * (1.0 - reduction_rate))))
        
        if target_k >= len(bg_units):
            kept_units.update(bg_units)
        else:
            sub_fps = fingerprints[bg_units]
            fps_safe = make_safe_fps(sub_fps)
            
            clusterer = AgglomerativeClustering(n_clusters=target_k, metric="euclidean", linkage="average")
            labels = clusterer.fit_predict(fps_safe)
            
            sub_scores = ablation_scores[bg_units]
            
            for c_id in range(target_k):
                members = np.where(labels == c_id)[0]
                if len(members) == 0:
                    continue
                medoid_sub_idx = find_causal_medoid(fps_safe, sub_scores, members)
                kept_units.add(bg_units[medoid_sub_idx])
                
    return sorted(list(kept_units))


def evaluate_masked_model(model, test_loader, kept_units, total_units, device):
    """Evaluates zero-shot performance by zeroing out unselected unit channels during inference."""
    model.eval()
    
    # Construct binary mask vector matching total units across all layers
    mask_vector = torch.zeros(total_units, device=device)
    mask_vector[list(kept_units)] = 1.0
    
    correct_zs = 0
    correct_agr = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            teacher_logits = model(x)
            teacher_preds = teacher_logits.argmax(dim=1)
            
            if hasattr(model, "forward_with_mask"):
                zs_logits = model.forward_with_mask(x, mask_vector)
            else:
                zs_logits = model(x)
                
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
    val_loader,
    test_loader,
    top_n_pct=15.0,
    device=torch.device("cpu"),
    cache_dir="cache",
    model_id="resnet18_cifar10",
    all_fps=None,
    ablation_scores=None,
    unit_map=None,
    **kwargs
):
    """
    Executes causal unit reservation and medoid clustering. Accepts pre-computed
    fingerprints and ablation scores if passed from main.py, or loads from/saves to cache.
    """
    # 1. Use passed arguments if provided; otherwise load from cache or compute
    if all_fps is not None and ablation_scores is not None and unit_map is not None:
        fingerprints = all_fps
    else:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{model_id}_causal_cache.pt")

        if os.path.exists(cache_path):
            print(f"--> [Cache Hit] Loading cached ablation scores & fingerprints from {cache_path}")
            cache_data = torch.load(cache_path, map_location="cpu")
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
            print(f"--> Saved causal metrics to {cache_path}")

    # Ensure numpy arrays for indexing and clustering
    if isinstance(ablation_scores, torch.Tensor):
        ablation_scores = ablation_scores.cpu().numpy()
    if isinstance(fingerprints, torch.Tensor):
        fingerprints = fingerprints.cpu().numpy()

    # 2. Hybrid Reserve Allocation & Background Medoid Clustering
    total_units = len(ablation_scores)
    k_reserve = max(1, int(round(total_units * (top_n_pct / 100.0))))
    
    sorted_unit_indices = np.argsort(ablation_scores)[::-1]
    reserved_units = set(sorted_unit_indices[:k_reserve])
    
    kept_units = cluster_background_units(
        fingerprints=fingerprints,
        ablation_scores=ablation_scores,
        unit_map=unit_map,
        reserved_units=reserved_units,
        reduction_rate=0.5
    )

    # 3. Evaluate zero-shot masked model
    sub_model = copy.deepcopy(teacher_model).to(device)
    zs_metrics = evaluate_masked_model(
        sub_model, test_loader, kept_units, total_units, device
    )

    return sub_model, kept_units, zs_metrics
