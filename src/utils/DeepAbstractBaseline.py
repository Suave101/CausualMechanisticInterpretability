# src/utils/DeepAbstractBaseline.py
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from src.utils.FineTuningMethod import fine_tune_submodel

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

def recalibrate_batchnorm(model, val_loader, mask_vector, device, num_batches=15):
    """Recalibrates BatchNorm running statistics after channel masking to prevent feature scale drift."""
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

def run_deepabstract_baseline(
    teacher_model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    train_loader: torch.utils.data.DataLoader = None,
    reduction_rate: float = 0.4,
    cluster_method: str = "kmeans",
    device: torch.device = torch.device("cpu"),
    ft_epochs: int = 0
):
    teacher_model.eval()
    
    val_imgs_list = []
    total_samples = 0
    for x, _ in val_loader:
        val_imgs_list.append(x)
        total_samples += x.size(0)
        if total_samples >= 256:
            break
    val_imgs = torch.cat(val_imgs_list)[:256].to(device)

    with torch.no_grad():
        activations = teacher_model.get_layer_activations(val_imgs)

    global_kept_indices = []
    global_unit_offset = 0
    total_orig_units = 0
    total_kept_units = 0

    for l_idx, act in enumerate(activations):
        if act.dim() == 4:
            fp = act.mean(dim=(2, 3)).T.cpu().numpy()
        else:
            fp = act.T.cpu().numpy()

        n_units = fp.shape[0]
        total_orig_units += n_units
        
        target_k = max(1, int(round(n_units * (1.0 - reduction_rate))))
        actual_k = min(target_k, n_units)
        
        fp_safe = make_safe_fps(fp)

        if actual_k == n_units:
            layer_kept = list(range(n_units))
        else:
            if cluster_method.lower() == "kmeans":
                clusterer = KMeans(n_clusters=actual_k, random_state=42, n_init=5)
            else:
                clusterer = AgglomerativeClustering(n_clusters=actual_k, metric="euclidean", linkage="average")
            
            labels = clusterer.fit_predict(fp_safe)
            layer_kept = []

            for c_id in range(actual_k):
                members = np.where(labels == c_id)[0]
                if len(members) == 0:
                    continue
                elif len(members) == 1:
                    layer_kept.append(members[0])
                else:
                    cluster_fps = fp_safe[members]
                    centroid = cluster_fps.mean(axis=0)
                    dists = np.linalg.norm(cluster_fps - centroid, axis=1)
                    medoid = members[np.argmin(dists)]
                    layer_kept.append(medoid)

        for u in layer_kept:
            global_kept_indices.append(global_unit_offset + u)
            
        global_unit_offset += n_units
        total_kept_units += len(layer_kept)

    # 1. Zero-Shot Evaluation Pass
    sub_model = copy.deepcopy(teacher_model).to(device)
    
    mask_vector = torch.zeros(total_orig_units, device=device)
    mask_vector[global_kept_indices] = 1.0

    # Recalibrate BatchNorm running statistics for the baseline model
    recalibrate_batchnorm(sub_model, val_loader, mask_vector, device)

    sub_model.eval()
    correct_zs = 0
    correct_agr = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            teacher_logits = teacher_model(x)
            teacher_preds = teacher_logits.argmax(dim=1)

            if hasattr(sub_model, "forward_with_mask"):
                abs_logits = sub_model.forward_with_mask(x, mask_vector)
            else:
                abs_logits = sub_model(x)

            abs_preds = abs_logits.argmax(dim=1)
            correct_zs += (abs_preds == y).sum().item()
            correct_agr += (abs_preds == teacher_preds).sum().item()
            total_samples += y.size(0)

    zs_acc = (correct_zs / total_samples) * 100.0
    zs_agreement = (correct_agr / total_samples) * 100.0

    print("\n--- [DeepAbstract Baseline Zero-Shot (Ashok et al. 2020)] ---")
    print(f"  -> Units Retained: {total_kept_units} / {total_orig_units} ({total_kept_units/total_orig_units*100:.1f}%)")
    print(f"  -> Zero-Shot Accuracy: {zs_acc:.2f}%")
    print(f"  -> Zero-Shot Agreement: {zs_agreement:.2f}%")

    ft_acc, ft_agreement = zs_acc, zs_agreement
    # 2. Fine-Tuning Pass (if ft_epochs > 0 and train_loader provided)
    if ft_epochs > 0 and train_loader is not None:
        ft_res = fine_tune_submodel(
            sub_model=sub_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            ft_epochs=ft_epochs
        )
        ft_acc = ft_res["ft_acc"]
        ft_agreement = ft_res["ft_agreement"]

    return {
        "method": "DeepAbstract_Baseline",
        "total_units": total_orig_units,
        "kept_units": total_kept_units,
        "zs_acc": zs_acc,
        "zs_agreement": zs_agreement,
        "ft_acc": ft_acc,
        "ft_agreement": ft_agreement
    }
