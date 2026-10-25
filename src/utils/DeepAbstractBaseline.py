import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering

def get_linear_layers(model: nn.Module):
    """Extracts all nn.Linear submodules from the model in execution order."""
    return [module for module in model.modules() if isinstance(module, nn.Linear)]

def evaluate_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device, mask_vector: torch.Tensor = None) -> float:
    """Evaluates classification accuracy of the network."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            if mask_vector is not None and hasattr(model, "forward_with_mask"):
                logits = model.forward_with_mask(x, mask_vector)
            else:
                logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0

def abstract_layer(
    sub_model: nn.Module,
    layer_idx: int,
    activations: np.ndarray,
    n_clusters: int,
    cluster_method: str,
    linear_layers: list,
    global_unit_offset: int
):
    """
    Applies Algorithm 1 from Ashok et al. (2020) to a single layer:
    1. Clusters raw activation vectors.
    2. Selects representative neurons (rep(C)) closest to centroids.
    3. Sums outgoing weights into representative neurons: W_{*, rep(C)}^{(l)} <- sum_{i in C} W_{*, i}^{(l)}.
    4. Zeroes out outgoing weights of non-representative neurons.
    """
    n_units = activations.shape[0]
    actual_k = min(max(1, n_clusters), n_units)
    
    # Cluster raw activation vectors directly (without L2-normalization)
    if actual_k == n_units:
        labels = np.arange(n_units)
    else:
        if cluster_method.lower() == "kmeans":
            clusterer = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        else:
            clusterer = AgglomerativeClustering(n_clusters=actual_k, metric="euclidean", linkage="average")
        labels = clusterer.fit_predict(activations)

    layer_kept = []
    layer_mask_indices = []

    # Identify linear layers connecting layer l to layer l+1
    has_linear = (linear_layers is not None and (layer_idx + 1) < len(linear_layers))
    next_linear = linear_layers[layer_idx + 1] if has_linear else None

    for c_id in range(actual_k):
        members = np.where(labels == c_id)[0]
        if len(members) == 0:
            continue
        
        # Find representative neuron rep(C) closest to cluster centroid
        if len(members) == 1:
            rep = members[0]
        else:
            cluster_fps = activations[members]
            centroid = cluster_fps.mean(axis=0)
            dists = np.linalg.norm(cluster_fps - centroid, axis=1)
            rep = members[np.argmin(dists)]

        layer_kept.append(rep)
        layer_mask_indices.append(global_unit_offset + rep)

        # Outgoing Weight Aggregation (Algorithm 1, Line 7)
        if next_linear is not None:
            with torch.no_grad():
                # Sum outgoing weights of all cluster members into representative
                summed_outgoing = next_linear.weight.data[:, members].sum(dim=1)
                next_linear.weight.data[:, rep] = summed_outgoing
                
                # Zero out outgoing weights for non-representatives
                non_reps = [m for m in members if m != rep]
                if len(non_reps) > 0:
                    next_linear.weight.data[:, non_reps] = 0.0

    return layer_kept, layer_mask_indices

def run_deepabstract_baseline(
    teacher_model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    reduction_rate: float = 0.4,
    accuracy_threshold: float = None,  # Parameter alpha in Paper Algorithm 2
    cluster_method: str = "kmeans",
    device: torch.device = torch.device("cpu"),
    num_calibration_samples: int = 54000,  # Calibration set size N
    train_loader: torch.utils.data.DataLoader = None,  # Accepted for main.py interface compatibility
    **kwargs
):
    """
    Executes the exact DeepAbstract framework (Ashok et al., 2020).
    """
    teacher_model.eval()

    # Safely batch-extract layer activations on GPU and aggregate on CPU RAM
    layer_fps_list = []
    total_samples = 0

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            batch_acts = teacher_model.get_layer_activations(x)
            
            if not layer_fps_list:
                layer_fps_list = [[] for _ in range(len(batch_acts))]
            
            for l_idx, act in enumerate(batch_acts):
                # Format 4D tensor (B, C, H, W) -> (C, B) or 2D tensor (B, C) -> (C, B)
                if act.dim() == 4:
                    act_pooled = act.mean(dim=(2, 3))
                else:
                    act_pooled = act
                layer_fps_list[l_idx].append(act_pooled.T.cpu().numpy())
            
            total_samples += x.size(0)
            if total_samples >= num_calibration_samples:
                break

    # Concatenate mini-batch activation fingerprints along the sample axis
    layer_activations = [
        np.concatenate(fps, axis=1)[:, :num_calibration_samples]
        for fps in layer_fps_list
    ]

    sub_model = copy.deepcopy(teacher_model).to(device)
    linear_layers = get_linear_layers(sub_model)

    global_kept_indices = []
    global_unit_offset = 0
    total_orig_units = 0
    total_kept_units = 0

    # Process hidden layers sequentially (Algorithm 1 & 2)
    for l_idx, fp in enumerate(layer_activations):
        n_units = fp.shape[0]
        total_orig_units += n_units
        
        # Target cluster count selection
        if accuracy_threshold is not None:
            # Algorithm 2: Binary Search for optimal k bounded by accuracy alpha
            low = 1
            high = n_units
            best_k = n_units
            
            while low <= high:
                mid = (low + high) // 2
                temp_model = copy.deepcopy(sub_model)
                temp_lineable = get_linear_layers(temp_model)
                
                abstract_layer(
                    temp_model, l_idx, fp, mid, cluster_method, 
                    temp_lineable, global_unit_offset
                )
                acc = evaluate_accuracy(temp_model, val_loader, device)
                
                if acc >= accuracy_threshold:
                    best_k = mid
                    high = mid - 1
                else:
                    low = mid + 1
            target_k = best_k
        else:
            target_k = max(1, int(round(n_units * (1.0 - reduction_rate))))

        # Apply abstraction to layer l
        layer_kept, mask_indices = abstract_layer(
            sub_model, l_idx, fp, target_k, cluster_method, 
            linear_layers, global_unit_offset
        )

        global_kept_indices.extend(mask_indices)
        global_unit_offset += n_units
        total_kept_units += len(layer_kept)

    # Construct global binary mask vector
    mask_vector = torch.zeros(total_orig_units, device=device)
    mask_vector[global_kept_indices] = 1.0

    # Pure Zero-Shot Evaluation Pass
    sub_model.eval()
    correct_zs = 0
    correct_agr = 0
    total_test_samples = 0

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
            total_test_samples += y.size(0)

    zs_acc = (correct_zs / total_test_samples) * 100.0
    zs_agreement = (correct_agr / total_test_samples) * 100.0

    print("\n--- [DeepAbstract Zero-Shot Abstraction (Ashok et al. 2020)] ---")
    print(f"  -> Units Retained: {total_kept_units} / {total_orig_units} ({total_kept_units/total_orig_units*100:.1f}%)")
    print(f"  -> Zero-Shot Accuracy: {zs_acc:.2f}%")
    print(f"  -> Teacher Agreement: {zs_agreement:.2f}%")

    return {
        "method": "DeepAbstract_Exact",
        "total_units": total_orig_units,
        "kept_units": total_kept_units,
        "zs_acc": zs_acc,
        "zs_agreement": zs_agreement
    }
