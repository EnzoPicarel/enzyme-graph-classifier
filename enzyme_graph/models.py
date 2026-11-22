"""Dense graph convolutions with graph-level mean pooling."""

import torch
from torch import nn
from torch.nn import functional as F

from .data import CLASS_COUNT, FEATURE_COUNT, NODE_TYPE_COUNT

ARCHITECTURES = ("gcn1", "gcn2", "gin")


class GraphClassifier(nn.Module):
    def __init__(self, architecture: str, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture: {architecture}")
        self.is_gin = architecture == "gin"
        units = FEATURE_COUNT + NODE_TYPE_COUNT
        depth = 1 if architecture == "gcn1" else 2
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if self.is_gin:
                layer = nn.Sequential(nn.Linear(units, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            else:
                layer = nn.Linear(units, hidden)
            self.layers.append(layer)
            units = hidden
        if self.is_gin:
            self.eps = nn.Parameter(torch.zeros(depth))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, CLASS_COUNT)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.is_gin:
            adjacency = adjacency + torch.diag_embed(mask.to(adjacency.dtype))
            inverse_degree = adjacency.sum(dim=-1).clamp_min(1).rsqrt()
            adjacency = inverse_degree.unsqueeze(1) * adjacency * inverse_degree.unsqueeze(2)
        node_mask = mask.unsqueeze(-1)
        for index, layer in enumerate(self.layers):
            aggregate = adjacency @ features
            if self.is_gin:
                aggregate = aggregate + (1 + self.eps[index]) * features
            features = F.relu(layer(aggregate)) * node_mask
        pooled = features.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.output(self.dropout(pooled))
