import pytest
import torch
import torch.nn as nn
from src.models.mlp import build_3layer_mlp, build_5layer_mlp


@pytest.fixture
def sample_batch():
    """Generates a dummy batch of 8 samples with feature dim 64."""
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    return x, y


def test_mlp3_output_shape(sample_batch):
    """Verify output tensor dimensions match (batch_size, num_classes)."""
    x, _ = sample_batch
    model = build_3layer_mlp(in_dim=64, num_classes=10)
    out = model(x)
    assert out.shape == (8, 10), f"Expected shape (8, 10), got {out.shape}"


def test_mlp_gradient_flow(sample_batch):
    """Verify backward pass populates non-zero, non-NaN gradients across all layers."""
    x, y = sample_batch
    model = build_3layer_mlp(in_dim=64, num_classes=10)
    
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN detected in gradient: {name}"
        assert torch.max(torch.abs(param.grad)) > 0, f"Zero gradient for parameter: {name}"


def test_layer_activations_extraction(sample_batch):
    """Verify get_layer_activations returns expected number of hidden layer outputs."""
    x, _ = sample_batch
    model = build_3layer_mlp(in_dim=64, num_classes=10)
    
    activations = model.get_layer_activations(x)
    
    # MLP3 has 3 hidden layers (128, 128, 64)
    assert len(activations) == 3, f"Expected 3 hidden activation tensors, got {len(activations)}"
    assert activations[0].shape == (8, 128)
    assert activations[1].shape == (8, 128)
    assert activations[2].shape == (8, 64)


def test_ablation_forward_pass(sample_batch):
    """Verify single-unit ablation alters output logits compared to baseline pass."""
    x, _ = sample_batch
    model = build_3layer_mlp(in_dim=64, num_classes=10)
    model.eval()

    with torch.no_grad():
        baseline_logits = model(x)
        # Ablate unit 0 in hidden layer 0
        ablated_logits = model.forward_with_ablation(x, layer_idx=0, unit_idx=0)

    assert not torch.allclose(baseline_logits, ablated_logits), \
        "Ablated logits are identical to baseline. Check forward_with_ablation logic."


def test_single_batch_overfit(sample_batch):
    """Sanity check to ensure model optimization capability (loss reduction on 1 batch)."""
    x, y = sample_batch
    model = build_3layer_mlp(in_dim=64, num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    initial_loss = nn.functional.cross_entropy(model(x), y).item()
    
    for _ in range(40):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

    final_loss = nn.functional.cross_entropy(model(x), y).item()
    assert final_loss < (initial_loss * 0.1), \
        f"Model failed to overfit single batch. Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}"
