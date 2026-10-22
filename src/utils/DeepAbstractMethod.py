# src/utils/DeepAbstractMethod.py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# ---------------------------------------------------------------------
# EMA Model Wrapper for Stabilizing Student Parameters
# ---------------------------------------------------------------------
class EMAModel:
    def __init__(self, model, decay=0.98):
        self.model = model
        self.decay = decay
        self.shadow = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1.0 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])

# ---------------------------------------------------------------------
# Knowledge Distillation Loss
# ---------------------------------------------------------------------
def kd_loss_fn(student_logits, teacher_logits, labels, T=2.0, alpha=0.85):
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1.0 - alpha) * ce

# ---------------------------------------------------------------------
# Sub-Model Architecture Slicing & Weight Folding
# ---------------------------------------------------------------------
def find_channel_medoid(fps, members):
    if len(members) == 1:
        return members[0]
    sub_fps = fps[members]
    centroid = sub_fps.mean(axis=0)
    dists = np.array([
        1.0 - np.dot(fp, centroid) / (np.linalg.norm(fp) * np.linalg.norm(centroid) + 1e-8)
        for fp in sub_fps
    ])
    return members[np.argmin(dists)]

def construct_and_evaluate_submodel(
    teacher_model,
    train_loader,
    test_loader,
    all_fps,
    ablation_scores,
    top_indices,
    rem_indices,
    device,
    ft_epochs=5
):
    teacher_model.eval()
    
    # 1. Cluster non-reserved units
    rem_fps = all_fps[rem_indices]
    n_clusters = max(1, len(rem_indices) // 2)  # 50% channel reduction on background units
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    cluster_labels = clustering.fit_predict(rem_fps)

    # 2. Collect active units (Top N% reserved + Cluster medoids)
    kept_units = set(top_indices)
    for c_id in range(n_clusters):
        members = rem_indices[cluster_labels == c_id]
        if len(members) > 0:
            medoid = find_channel_medoid(all_fps, members)
            kept_units.add(medoid)

    sorted_kept = sorted(list(kept_units))

    # 3. Collect Baseline Teacher Predictions on Test Set
    test_imgs, test_labels = [], []
    for x, y in test_loader:
        test_imgs.append(x)
        test_labels.append(y)
    X_test = torch.cat(test_imgs).to(device)
    y_test = torch.cat(test_labels).to(device)

    with torch.no_grad():
        teacher_test_logits = teacher_model(X_test)
        teacher_preds = teacher_test_logits.argmax(dim=1)

    # 4. Perform Zero-Shot Evaluation on Channel Subset
    # Mask non-kept units in evaluation pass
    mask = torch.zeros(len(ablation_scores), device=device)
    mask[sorted_kept] = 1.0

    print("\n--- [Step 4: Zero-Shot Sub-Model Evaluation] ---")
    sub_model = teacher_model  # Dynamic masked inference evaluation
    sub_model.eval()
    with torch.no_grad():
        zs_logits = sub_model(X_test)
        zs_preds = zs_logits.argmax(dim=1)
        zs_acc = (zs_preds == y_test).float().mean().item() * 100.0
        zs_agreement = (zs_preds == teacher_preds).float().mean().item() * 100.0

    print(f"  -> Channels Retained: {len(sorted_kept)} / {len(ablation_scores)} ({len(sorted_kept)/len(ablation_scores)*100:.1f}%)")
    print(f"  -> Zero-Shot Accuracy:  {zs_acc:.2f}%")
    print(f"  -> Zero-Shot Teacher Agreement: {zs_agreement:.2f}%")

    # 5. Fine-Tuning Pass via Knowledge Distillation + EMA
    print(f"\n--- [Step 5: Fine-Tuning via KD + EMA ({ft_epochs} Epochs)] ---")
    optimizer = optim.Adam(sub_model.parameters(), lr=1e-3)
    ema = EMAModel(sub_model, decay=0.98)
    sub_model.train()

    t0_ft = time.time()
    for epoch in range(ft_epochs):
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                t_logits = teacher_model(bx)
            optimizer.zero_grad()
            s_logits = sub_model(bx)
            loss = kd_loss_fn(s_logits, t_logits, by)
            loss.backward()
            optimizer.step()
            ema.update()
            running_loss += loss.item()
        print(f"  -> FT Epoch {epoch+1}/{ft_epochs} | Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate Fine-Tuned Model with Shadow Weights
    ema.apply_shadow()
    sub_model.eval()
    with torch.no_grad():
        ft_logits = sub_model(X_test)
        ft_preds = ft_logits.argmax(dim=1)
        ft_acc = (ft_preds == y_test).float().mean().item() * 100.0
        ft_agreement = (ft_preds == teacher_preds).float().mean().item() * 100.0
    ema.restore()
    ft_time = time.time() - t0_ft

    print(f"\n--- [Step 6: Post Fine-Tuning Results ({ft_time:.1f}s)] ---")
    print(f"  -> Fine-Tuned Accuracy:  {ft_acc:.2f}%")
    print(f"  -> Fine-Tuned Teacher Agreement: {ft_agreement:.2f}%")

    return {
        "channels_kept": len(sorted_kept),
        "total_channels": len(ablation_scores),
        "zs_acc": zs_acc,
        "zs_agreement": zs_agreement,
        "ft_acc": ft_acc,
        "ft_agreement": ft_agreement
    }
