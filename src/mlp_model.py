from __future__ import annotations

import torch
from torch import nn


class BiofilmMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            # 所有隐层都加 Dropout（含最后一层）
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LegacyBiofilmMLP(nn.Module):
    """Compatibility loader for checkpoints saved before BatchNorm was added."""

    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
