# src/utils/CausalRestructuringMethod.py
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def make_safe_fps(fps):
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

def evaluate_masked_model(model, X_test, kept_units, total_units):
    """Evaluates model by zeroing out activations for unselected units."""
    model.eval()
    
    # Construct binary mask over all global unit indices
    mask_vector = torch.zeros(total_units, device=X_test.device)
    mask_vector[list(kept_units)] = 1.0
    
    with torch.no_grad():
        if hasattr(model, "forward_with_mask"):
            logits = model.forward_with_mask(X_test, mask_vector)
        else:
            logits = model(X_test)
            
    return logits

def perform_causal_restructuring(
    teacher_model,
    all_fps,
    ablation_scores,
    top_indices,
    rem_indices,
    test_loader,
    device
):
    teacher_model.eval()
    
    # 1. Cluster non-reserved background units
    rem_fps_safe = make_safe_fps(all_fps[rem_indices])
    n_clusters = max(1, len(rem_indices) // 2)
    actual_clusters = min(n_clusters, len(rem_indices))
    
    clustering = AgglomerativeClustering(n_clusters=actual_clusters, metric="euclidean", linkage="average")
    cluster_labels = clustering.fit_predict(rem_fps_safe)

    # 2. Select Global Causal Medoids
    kept_units = set(top_indices)
    for c_id in range(actual_clusters):
        members = rem_indices[cluster_labels == c_id]
        if len(members) > 0:
            medoid = find_causal_medoid(all_fps, ablation_scores, members)
            kept_units.add(medoid)

    sorted_kept = sorted(list(kept_units), key=lambda idx: ablation_scores[idx], reverse=True)

    # 3. Deep Copy Teacher to isolate Student Model
    sub_model = copy.deepcopy(teacher_model).to(device)

    # 4. Collect Test Set Baseline Targets
    test_imgs, test_labels = [], []
    for x, y in test_loader:
        test_imgs.append(x)
        test_labels.append(y)
    X_test = torch.cat(test_imgs).to(device)
    y_test = torch.cat(test_labels).to(device)

    with torch.no_grad():
        teacher_test_logits = teacher_model(X_test)
        teacher_preds = teacher_test_logits.argmax(dim=1)

    # 5. Zero-Shot Sub-Model Evaluation
    print("\n--- [Step 4: Zero-Shot Sub-Model Evaluation] ---")
    zs_logits = evaluate_masked_model(sub_model, X_test, kept_units, len(ablation_scores))
    zs_preds = zs_logits.argmax(dim=1)
    zs_acc = (zs_preds == y_test).float().mean().item() * 100.0
    zs_agreement = (zs_preds == teacher_preds).float().mean().item() * 100.0

    print(f"  -> Retained Units (Globally Ranked): {len(sorted_kept)} / {len(ablation_scores)} ({len(sorted_kept)/len(ablation_scores)*100:.1f}%)")
    print(f"  -> Zero-Shot Accuracy:  {zs_acc:.2f}%")
    print(f"  -> Zero-Shot Teacher Agreement: {zs_agreement:.2f}%")

    return sub_model, sorted_kept, {
        "channels_kept": len(sorted_kept),
        "total_channels": len(ablation_scores),
        "zs_acc": zs_acc,
        "zs_agreement": zs_agreement
    }