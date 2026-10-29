# src/utils/CausalRestructuringMethod.py
import copy
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.utils.CMI_Method import compute_causal_fingerprints_and_scores, compute_activation_fingerprints


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


def find_causal_medoid(fps, ablation_scores, members, causal_weight=0.5, activity=None, dead_tol=1e-6):
    """
    Selects cluster medoid balancing fingerprint distance and causal score.
    causal_weight (alpha) in [0.0, 1.0]:
      - 0.0 = Pure geometric centroid distance
      - 1.0 = Pure max causal score selection
    activity: raw (un-normalized) fingerprint norms; dead/inactive channels are never chosen
    when an active member exists.
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

    if activity is not None:
        sub_activity = activity[members]
        dead = sub_activity <= dead_tol * (np.max(sub_activity) + 1e-12)
        if not np.all(dead):
            combined_score = combined_score + dead * 1e6

    return members[np.argmin(combined_score)]


def cluster_background_units(
    fingerprints, 
    ablation_scores, 
    unit_map, 
    reserved_units, 
    reduction_rate=0.5,
    causal_weight=0.5,
    dead_rel_tol=1e-6
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

        # reduction_rate may be a float (all layers) or a {layer_idx: rate} dict
        layer_rr = reduction_rate[l_idx] if isinstance(reduction_rate, dict) else reduction_rate
        target_k = int(round(len(bg_units) * (1.0 - layer_rr)))
        if target_k <= 0:
            if len(layer_cluster_map[l_idx]) > 0:
                continue
            target_k = 1

        # Dead units (no activation on any profiled sample) carry no signal: never spend budget on
        # them. Otherwise each gets an arbitrary direction from make_safe_fps and its own cluster.
        layer_activity = np.linalg.norm(fingerprints[layer_global_indices], axis=1)
        dead_tol = dead_rel_tol * (layer_activity.max() + 1e-12)
        alive_bg = [g_idx for g_idx in bg_units if np.linalg.norm(fingerprints[g_idx]) > dead_tol]
        if alive_bg:
            bg_units = alive_bg

        if target_k >= len(bg_units):
            for g_idx in bg_units:
                kept_units.add(g_idx)
                u_idx = unit_map[g_idx][1]
                layer_cluster_map[l_idx].append((u_idx, [u_idx]))
        else:
            sub_fps = fingerprints[bg_units]
            fps_safe = make_safe_fps(sub_fps)
            sub_activity = np.linalg.norm(sub_fps, axis=1)

            clusterer = AgglomerativeClustering(n_clusters=target_k, metric="cosine", linkage="average")
            labels = clusterer.fit_predict(fps_safe)
            sub_scores = ablation_scores[bg_units]
            
            for c_id in range(target_k):
                members_sub_idx = np.where(labels == c_id)[0]
                if len(members_sub_idx) == 0:
                    continue
                
                medoid_sub_idx = find_causal_medoid(
                    fps_safe, sub_scores, members_sub_idx, causal_weight=causal_weight,
                    activity=sub_activity
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


def extract_target_bns(model: nn.Module, param_layers):
    """
    Returns the BatchNorm applied directly to each target layer's output (or None).
    Convention: a conv named '<prefix>conv<k>' is normalized by '<prefix>bn<k>'.
    """
    modules = dict(model.named_modules())
    names = {id(m): n for n, m in modules.items()}
    bns = []
    for layer in param_layers:
        prefix, _, leaf = names.get(id(layer), "").rpartition(".")
        if not leaf.startswith("conv"):
            bns.append(None)
            continue
        bn_leaf = leaf.replace("conv", "bn", 1)
        bn = modules.get(f"{prefix}.{bn_leaf}" if prefix else bn_leaf)
        bns.append(bn if isinstance(bn, nn.modules.batchnorm._BatchNorm) else None)
    return bns


def extract_foldable_layers(model: nn.Module, num_layers: int):
    """
    Indices of target layers whose output feeds ONLY the next target layer. Outputs that
    enter a residual addition are not foldable: the next conv sees relu(out + shortcut).
    """
    if hasattr(model, "get_foldable_layers") and callable(model.get_foldable_layers):
        return set(model.get_foldable_layers())
    return set(range(num_layers))


def apply_step4_weight_merging(model: nn.Module, layer_cluster_map: dict):
    """
    Executes Step 4 Weight Merging:
    1. Incoming Weight Averaging (weights, bias, and BN affine parameters)
    2. Outgoing Weight Folding scaled by sqrt(|m|)
    3. Zeroing out non-representative weights
    Only applied where the next target layer directly consumes this layer's output; for
    residual-stream outputs the non-medoids are removed by the activation mask alone.
    """
    param_layers = extract_target_layers(model)
    param_bns = extract_target_bns(model, param_layers)
    foldable = extract_foldable_layers(model, len(param_layers))

    with torch.no_grad():
        for l_idx, clusters in layer_cluster_map.items():
            if l_idx >= len(param_layers):
                continue
                
            if l_idx not in foldable:
                continue

            curr_layer = param_layers[l_idx]
            curr_bn = param_bns[l_idx]
            has_next= (l_idx + 1) < len(param_layers)
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
                if curr_bn is not None and curr_bn.affine:
                    curr_bn.weight.data[medoid_u] = curr_bn.weight.data[members_u].mean(dim=0)
                    curr_bn.bias.data[medoid_u] = curr_bn.bias.data[members_u].mean(dim=0)

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


def recalibrate_batchnorm(model, val_loader, mask_vector, device, num_batches=50):
    """Recalibrates BatchNorm statistics post-restructuring (fresh cumulative average)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    saved_momentum = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
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
    for bn, momentum in zip(bns, saved_momentum):
        bn.momentum = momentum
    model.eval()


def collect_activation_stats(model, data_loader, device, max_samples=2048):
    """
    Per-layer channel mean and (centered) covariance of the profiled activations, pooled over
    samples and spatial positions. Returns {layer_idx: (mean[C], cov[C, C])} in float64.
    """
    model.eval()
    sums, outer, counts = {}, {}, {}
    seen = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x[: max_samples - seen].to(device)
            for l_idx, act in enumerate(model.get_layer_activations(x)):
                a = act.permute(0, 2, 3, 1).reshape(-1, act.shape[1]) if act.dim() == 4 else act
                a = a.double()
                if l_idx not in sums:
                    sums[l_idx] = torch.zeros(a.shape[1], dtype=torch.float64, device=device)
                    outer[l_idx] = torch.zeros(a.shape[1], a.shape[1], dtype=torch.float64, device=device)
                    counts[l_idx] = 0
                sums[l_idx] += a.sum(dim=0)
                outer[l_idx] += a.T @ a
                counts[l_idx] += a.shape[0]
            seen += x.shape[0]
            if seen >= max_samples:
                break

    stats = {}
    for l_idx in sums:
        mean = sums[l_idx] / counts[l_idx]
        cov = outer[l_idx] / counts[l_idx] - torch.outer(mean, mean)
        stats[l_idx] = (mean, cov)
    return stats


def apply_lsq_reconstruction(model: nn.Module, layer_cluster_map: dict, act_stats: dict, ridge=1e-3):
    """
    Least-squares alternative to Step 4 folding. Every channel h of a layer is regressed onto
    the kept (medoid/reserved) channels, h ~= A h_K + c, and the next layer's input weights
    become W' = W A (non-kept columns zeroed), so W h ~= W' h_K + W c. Kept channels keep their
    original incoming weights. The constant W c goes into the next layer's bias when it has
    one; otherwise (conv -> BN) BatchNorm recalibration absorbs it.
    """
    param_layers = extract_target_layers(model)
    foldable = extract_foldable_layers(model, len(param_layers))

    with torch.no_grad():
        for l_idx, clusters in layer_cluster_map.items():
            if l_idx not in foldable or l_idx + 1 >= len(param_layers) or l_idx not in act_stats:
                continue
            next_layer = param_layers[l_idx + 1]
            mean, cov = act_stats[l_idx]
            n_channels = mean.shape[0]
            if next_layer.weight.shape[1] != n_channels:
                continue

            kept = sorted(medoid for medoid, _ in clusters)
            dropped = sorted(set(range(n_channels)) - set(kept))
            if not dropped:
                continue

            cov = cov.to(next_layer.weight.device)
            mean = mean.to(next_layer.weight.device)
            K = torch.tensor(kept, device=cov.device)
            cov_kk = cov[K][:, K]
            lam = ridge * torch.trace(cov_kk) / len(kept) + 1e-12
            A = cov[:, K] @ torch.linalg.inv(cov_kk + lam * torch.eye(len(kept), dtype=cov.dtype, device=cov.device))
            offset = mean - A @ mean[K]

            W = next_layer.weight.data.double()
            new_W = torch.zeros_like(W)
            if W.dim() == 2:
                new_W[:, K] = W @ A
                shift = W @ offset
            else:
                new_W[:, K] = torch.einsum("oihw,ik->okhw", W, A)
                shift = torch.einsum("oihw,i->o", W, offset)
            next_layer.weight.data.copy_(new_W.to(next_layer.weight.dtype))
            if next_layer.bias is not None:
                next_layer.bias.data += shift.to(next_layer.bias.dtype)

            curr_layer = param_layers[l_idx]
            curr_layer.weight.data[dropped] = 0.0
            if curr_layer.bias is not None:
                curr_layer.bias.data[dropped] = 0.0


def layer_unit_groups(unit_map):
    """{layer_idx: [global unit indices]} in layer order."""
    groups = {}
    for global_idx, (l_idx, _) in enumerate(unit_map):
        groups.setdefault(l_idx, []).append(global_idx)
    return groups


def select_layer_reserves(ablation_scores, layer_groups, layer_reserve_counts):
    """Top-scoring units of each layer, with layer_reserve_counts[l] units per layer."""
    reserved = set()
    for l_idx, indices in layer_groups.items():
        indices = np.array(indices)
        order = np.argsort(ablation_scores[indices])[::-1]
        reserved.update(indices[order[: layer_reserve_counts[l_idx]]].tolist())
    return reserved


def layer_plan_for_keep_ratio(n_units, keep_ratio, reserve_share):
    """(reserve count, background reduction rate) that keeps ~keep_ratio of a layer's units."""
    n_keep = min(n_units, max(1, int(round(keep_ratio * n_units))))
    n_res = min(n_keep, max(1, int(round(reserve_share * n_keep))))
    rr = 1.0 - (n_keep - n_res) / (n_units - n_res) if n_units > n_res else 0.0
    return n_res, rr


def build_restructured_model(teacher_model, layer_cluster_map, fold_mode, act_stats, device):
    sub_model = copy.deepcopy(teacher_model).to(device)
    if fold_mode == "lsq":
        apply_lsq_reconstruction(sub_model, layer_cluster_map, act_stats)
    else:
        apply_step4_weight_merging(sub_model, layer_cluster_map)
    return sub_model


def profile_layer_sensitivity(
    teacher_model,
    fingerprints,
    ablation_scores,
    unit_map,
    calib_loader,
    device,
    reserve_share=0.3,
    causal_weight=0.5,
    fold_mode="sqrt",
    act_stats=None,
    keep_grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    num_samples=512,
    recal_batches=10,
):
    """
    Restructures one layer at a time (all others intact) at each keep ratio in keep_grid and
    records KL(teacher || sub-model) on calibration data.
    Returns {layer_idx: {keep_ratio: kl}}.
    """
    groups = layer_unit_groups(unit_map)
    total_units = len(unit_map)

    xs, seen = [], 0
    for x, _ in calib_loader:
        xs.append(x)
        seen += x.shape[0]
        if seen >= num_samples:
            break
    calib_x = torch.cat(xs)[:num_samples].to(device)
    teacher_model.eval()
    with torch.no_grad():
        teacher_logp = torch.log_softmax(teacher_model(calib_x).float(), dim=1)

    sensitivity = {}
    for l_idx, indices in groups.items():
        sensitivity[l_idx] = {}
        others = set(range(total_units)) - set(indices)
        for keep_ratio in keep_grid:
            if keep_ratio >= 1.0:
                sensitivity[l_idx][keep_ratio] = 0.0
                continue
            n_res, rr = layer_plan_for_keep_ratio(len(indices), keep_ratio, reserve_share)
            reserved = select_layer_reserves(ablation_scores, {l_idx: indices}, {l_idx: n_res}) | others
            kept, cmap = cluster_background_units(
                fingerprints, ablation_scores, unit_map, reserved,
                reduction_rate={l: (rr if l == l_idx else 0.0) for l in groups},
                causal_weight=causal_weight,
            )
            sub_model = build_restructured_model(teacher_model, {l_idx: cmap[l_idx]}, fold_mode, act_stats, device)
            mask = torch.zeros(total_units, device=device)
            mask[kept] = 1.0
            recalibrate_batchnorm(sub_model, calib_loader, mask, device, num_batches=recal_batches)
            with torch.no_grad():
                sub_logp = torch.log_softmax(sub_model.forward_with_mask(calib_x, mask).float(), dim=1)
                kl = torch.sum(teacher_logp.exp() * (teacher_logp - sub_logp), dim=1).mean().item()
            sensitivity[l_idx][keep_ratio] = kl
    return sensitivity


def allocate_layer_budget(sensitivity, layer_sizes, keep_fraction):
    """
    Greedy budget allocation: starting from every layer fully kept, repeatedly lower the keep
    ratio of the layer with the smallest KL increase per unit removed, until the total number
    of kept units reaches keep_fraction. Returns {layer_idx: keep_ratio}.
    """
    total = sum(layer_sizes.values())
    target = keep_fraction * total

    # monotone loss curves (KL can only grow as more units are removed)
    curves = {}
    for l_idx, losses in sensitivity.items():
        grid = sorted(losses.keys(), reverse=True)
        running, curve = 0.0, []
        for r in grid:
            running = max(running, losses[r])
            curve.append((r, running))
        curves[l_idx] = curve

    position = {l_idx: 0 for l_idx in curves}
    keep = {l_idx: 1.0 for l_idx in curves}
    kept_units = float(total)
    while kept_units > target:
        best, best_cost = None, None
        for l_idx, curve in curves.items():
            p = position[l_idx]
            if p + 1 >= len(curve):
                continue
            (r0, loss0), (r1, loss1) = curve[p], curve[p + 1]
            removed = (r0 - r1) * layer_sizes[l_idx]
            if removed <= 0:
                continue
            cost = (loss1 - loss0) / removed
            if best_cost is None or cost < best_cost:
                best, best_cost = l_idx, cost
        if best is None:
            break
        p = position[best]
        r0, r1 = curves[best][p][0], curves[best][p + 1][0]
        removed = (r0 - r1) * layer_sizes[best]
        if kept_units - removed < target:
            keep[best] = r0 - (kept_units - target) / layer_sizes[best]
            kept_units = target
        else:
            keep[best] = r1
            position[best] = p + 1
            kept_units -= removed
    return keep


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
    reserve_scope="layer",
    fold_mode="sqrt",
    budget_mode="uniform",
    calib_loader=None,
    fp_samples=None,
    fp_pool=1,
    lsq_samples=2048,
    act_stats=None,
    layer_sensitivity=None,
    **kwargs
):
    """
    fold_mode:   "sqrt" = doc Step 4 (average incoming, fold outgoing / sqrt|m|);
                 "lsq"  = least-squares rebuild of the next layer from kept channels.
    budget_mode: "uniform" = same reduction rate in every layer;
                 "sensitivity" = same total budget, allocated by per-layer KL sensitivity.
    fp_samples / fp_pool: recompute fingerprints on calib_loader with more samples and/or a
                 fp_pool x fp_pool spatial grid instead of the global spatial mean.
    act_stats / layer_sensitivity: precomputed results to reuse across calls.
    """
    if test_loader is None and val_loader is not None:
        test_loader = val_loader
    if val_loader is None and test_loader is not None:
        val_loader = test_loader

    if all_fps is not None and ablation_scores is not None and unit_map is not None:
        fingerprints = all_fps
    else:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{model_id}_causal_cache_fp32.pt")

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
    calib_loader = calib_loader if calib_loader is not None else val_loader

    # Optional richer fingerprints: more samples and/or a spatial pooling grid
    if fp_samples is not None or fp_pool > 1:
        fingerprints = compute_activation_fingerprints(
            teacher_model, calib_loader, device, max_samples=fp_samples or 256, pool=fp_pool
        )
    if isinstance(fingerprints, torch.Tensor):
        fingerprints = fingerprints.cpu().numpy()

    total_units = len(ablation_scores)
    groups = layer_unit_groups(unit_map)

    if act_stats is None and fold_mode == "lsq":
        act_stats = collect_activation_stats(teacher_model, calib_loader, device, max_samples=lsq_samples)

    if budget_mode == "sensitivity":
        # Same total budget as the uniform plan, distributed by measured layer sensitivity
        keep_fraction = (top_n_pct + (100.0 - top_n_pct) * (1.0 - reduction_rate)) / 100.0
        reserve_share = (top_n_pct / 100.0) / keep_fraction if keep_fraction > 0 else 0.0
        if layer_sensitivity is None:
            layer_sensitivity = profile_layer_sensitivity(
                teacher_model, fingerprints, ablation_scores, unit_map, calib_loader, device,
                reserve_share=reserve_share, causal_weight=causal_weight,
                fold_mode=fold_mode, act_stats=act_stats,
            )
        layer_keep = allocate_layer_budget(
            layer_sensitivity, {l: len(g) for l, g in groups.items()}, keep_fraction
        )
        plans = {l: layer_plan_for_keep_ratio(len(groups[l]), layer_keep[l], reserve_share) for l in groups}
        reserved_units = select_layer_reserves(ablation_scores, groups, {l: p[0] for l, p in plans.items()})
        layer_reduction = {l: p[1] for l, p in plans.items()}
    else:
        # reserve_scope="layer": top r% within each layer (doc Step 2 layer-wise ranking);
        # reserve_scope="global": top r% across all units
        reserve_groups = {"all": list(range(total_units))} if reserve_scope == "global" else groups
        reserved_units = select_layer_reserves(
            ablation_scores, reserve_groups,
            {l: max(1, int(round(len(g) * (top_n_pct / 100.0)))) for l, g in reserve_groups.items()},
        )
        layer_reduction = reduction_rate

    kept_units, layer_cluster_map = cluster_background_units(
        fingerprints=fingerprints,
        ablation_scores=ablation_scores,
        unit_map=unit_map,
        reserved_units=reserved_units,
        reduction_rate=layer_reduction,
        causal_weight=causal_weight
    )

    mask_vector = torch.zeros(total_units, device=device)
    mask_vector[kept_units] = 1.0

    sub_model = build_restructured_model(teacher_model, layer_cluster_map, fold_mode, act_stats, device)

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
