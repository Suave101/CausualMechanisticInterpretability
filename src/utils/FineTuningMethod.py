# src/utils/FineTuningMethod.py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EMAModel:
    """Exponential Moving Average (EMA) Parameter Smoother."""
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

def kd_loss_fn(student_logits, teacher_logits, labels, T=2.0, alpha=0.85):
    """Knowledge Distillation Loss combining soft KL-Divergence and Cross-Entropy."""
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    kl = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1.0 - alpha) * ce

def fine_tune_submodel(
    sub_model,
    teacher_model,
    train_loader,
    test_loader,
    device,
    ft_epochs=5,
    lr=1e-3,
    beta=0.98,
    T=2.0,
    alpha=0.85
):
    """
    Executes Knowledge Distillation (KD) fine-tuning with Exponential Moving Average (EMA).
    """
    print(f"\n--- [Step 5: Fine-Tuning via KD + EMA ({ft_epochs} Epochs)] ---")
    optimizer = optim.Adam(sub_model.parameters(), lr=lr)
    ema = EMAModel(sub_model, decay=beta)
    
    teacher_model.eval()
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
            loss = kd_loss_fn(s_logits, t_logits, by, T=T, alpha=alpha)
            loss.backward()
            optimizer.step()
            ema.update()
            running_loss += loss.item()
        print(f"  -> FT Epoch {epoch+1}/{ft_epochs} | Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate Fine-Tuned Model using EMA Shadow Weights
    test_imgs, test_labels = [], []
    for x, y in test_loader:
        test_imgs.append(x)
        test_labels.append(y)
    X_test = torch.cat(test_imgs).to(device)
    y_test = torch.cat(test_labels).to(device)

    with torch.no_grad():
        teacher_preds = teacher_model(X_test).argmax(dim=1)

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
        "ft_acc": ft_acc,
        "ft_agreement": ft_agreement,
        "ft_time": ft_time
    }
