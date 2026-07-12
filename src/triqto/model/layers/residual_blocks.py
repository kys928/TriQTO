"""Small residual MLP blocks shared across Phase 13 modules."""
from __future__ import annotations

from torch import Tensor, nn


class ResidualMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        layers: int = 2,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or layers <= 0:
            raise ValueError("hidden_dim and layers must be positive")
        blocks: list[nn.Module] = []
        for _ in range(layers):
            blocks.extend(
                (
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
            )
        self.layers = nn.ModuleList(blocks)
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
            for _ in range(layers)
        )
        self.layers_per_block = 4

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for index, norm in enumerate(self.norms):
            offset = index * self.layers_per_block
            update = hidden
            for layer in self.layers[offset : offset + self.layers_per_block]:
                update = layer(update)
            hidden = norm(hidden + update)
        return hidden


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


__all__ = ["ProjectionMLP", "ResidualMLP"]
