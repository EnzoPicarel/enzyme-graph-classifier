"""Dense graph convolutions with graph-level mean pooling."""

import torch
from torch import nn
from torch.nn import functional as F

from .data import CLASS_COUNT, FEATURE_COUNT, NODE_TYPE_COUNT

BASELINES = ("gcn1", "gcn2", "gin")
MESSAGE_PASSING = ("mpnn3", "mpnn3-sum", "mpnn1")
ARCHITECTURES = BASELINES + MESSAGE_PASSING


class GraphClassifier(nn.Module):
    def __init__(self, architecture: str, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        if architecture not in BASELINES:
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


class ResidualGraphClassifier(nn.Module):
    """Pool graph-wide feature moments before and after each message-passing layer."""

    def __init__(self, architecture: str, hidden: int = 96):
        super().__init__()
        self.sum_neighbors = architecture == "mpnn3-sum"
        depth = 1 if architecture == "mpnn1" else 3
        self.embed = nn.Linear(FEATURE_COUNT + NODE_TYPE_COUNT + 1, hidden)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden),
                          nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            for _ in range(depth)
        )
        readout_width = (depth + 1) * hidden * 3 + 1
        self.classifier = nn.Sequential(
            nn.LayerNorm(readout_width), nn.Linear(readout_width, hidden * 2),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden * 2, CLASS_COUNT),
        )

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        node_mask = mask.unsqueeze(-1)
        size = node_mask.sum(dim=1)
        degree = adjacency.sum(dim=-1, keepdim=True)
        features = torch.cat((features.clamp(-6, 6), degree.log1p()), dim=-1)
        features = F.gelu(self.embed(features)) * node_mask
        pooled = list(self.pool(features, node_mask, size))
        for layer in self.layers:
            neighbors = adjacency @ features
            if not self.sum_neighbors:
                neighbors = neighbors / degree.clamp_min(1)
            update = layer(torch.cat((features, neighbors), dim=-1))
            features = (features + update) * node_mask
            pooled.extend(self.pool(features, node_mask, size))
        return self.classifier(torch.cat((*pooled, size.float().log1p()), dim=-1))

    @staticmethod
    def pool(features: torch.Tensor, node_mask: torch.Tensor, size: torch.Tensor) -> tuple:
        mean = features.sum(dim=1) / size
        std = (features.square().sum(dim=1) / size - mean.square()).clamp_min(1e-6).sqrt()
        maximum = features.masked_fill(~node_mask, -torch.inf).amax(dim=1)
        return mean, std, maximum


def make_model(architecture: str) -> nn.Module:
    if architecture in MESSAGE_PASSING:
        return ResidualGraphClassifier(architecture)
    return GraphClassifier(architecture)
