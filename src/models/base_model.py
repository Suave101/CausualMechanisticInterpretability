# src/models/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class RearrangeableModel(nn.Module, ABC):
    """Abstract interface for models whose internal neurons/channels can be causally analyzed and rearranged."""
    
    @abstractmethod
    def get_layer_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract hidden activation tensors for fingerprinting."""
        pass

    @abstractmethod
    def forward_with_ablation(self, x: torch.Tensor, layer_idx: int, unit_idx: int) -> torch.Tensor:
        """Run forward pass with neuron/channel (layer_idx, unit_idx) set to 0."""
        pass

    @abstractmethod
    def get_total_units(self) -> int:
        """Return total count of rearrangeable units (FC neurons or Conv channels)."""
        pass
