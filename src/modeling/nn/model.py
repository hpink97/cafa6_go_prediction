"""The multi-label network: per-feature-group projections, an optional fused promoter branch, a
shared MLP trunk, and one logit per GO term; plus its losses."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEFAULT_PROJECTION_DIMS = {
    "esm2": 256,
    "promoter_nt": 128,
    "promoter_agront": 128,
    "manual_taxonomy": 32,
}
PROMOTER_FUSION_DIM = 128


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def feature_group_indices(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {name: [] for name in DEFAULT_PROJECTION_DIMS}
    for index, feature_name in enumerate(feature_names):
        if feature_name.startswith("esm2_"):
            group = "esm2"
        elif feature_name.startswith("promoter_nt_"):
            group = "promoter_nt"
        elif feature_name.startswith("promoter_agront_"):
            group = "promoter_agront"
        else:
            group = "manual_taxonomy"
        groups[group].append(index)
    return {name: indices for name, indices in groups.items() if indices}


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP layer: x + dropout(GELU(linear(layernorm(x))))."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layer(features)


class MultilabelGOModel(nn.Module):
    """Project each feature modality, then share a compact prediction trunk."""

    def __init__(
        self,
        feature_groups: dict[str, list[int]],
        n_terms: int,
        trunk_dim: int = 512,
        bottleneck_dim: int = 256,
        branch_dropout: float = 0.1,
        trunk_dropout: float = 0.2,
        projection_dims: dict[str, int] | None = None,
        promoter_fusion_dim: int = PROMOTER_FUSION_DIM,
        trunk_layers: int = 2,
        trunk_residual: bool = False,
    ) -> None:
        super().__init__()
        if trunk_layers < 2:
            raise ValueError("trunk_layers must be at least 2 (input and bottleneck layers)")
        projection_dims = {**DEFAULT_PROJECTION_DIMS, **(projection_dims or {})}
        self.group_names = tuple(feature_groups)
        self.promoter_group_names = tuple(name for name in self.group_names if name.startswith("promoter_"))
        self.non_promoter_group_names = tuple(name for name in self.group_names if not name.startswith("promoter_"))
        self.branches = nn.ModuleDict()
        non_promoter_dim = 0
        promoter_dim = 0

        for group_name, indices in feature_groups.items():
            projection_dim = projection_dims[group_name]
            buffer_name = f"{group_name}_indices"
            self.register_buffer(buffer_name, torch.tensor(indices, dtype=torch.long), persistent=False)
            self.branches[group_name] = nn.Sequential(
                nn.LayerNorm(len(indices)),
                nn.Linear(len(indices), projection_dim),
                nn.GELU(),
                nn.Dropout(branch_dropout),
            )
            if group_name.startswith("promoter_"):
                promoter_dim += projection_dim
            else:
                non_promoter_dim += projection_dim

        self.promoter_fusion: nn.Module | None = None
        if promoter_dim:
            self.promoter_fusion = nn.Sequential(
                nn.Linear(promoter_dim, promoter_fusion_dim),
                nn.GELU(),
                nn.Dropout(branch_dropout),
            )
        fused_dim = non_promoter_dim + (promoter_fusion_dim if promoter_dim else 0)

        # Two layers keep the original module layout, so older checkpoints still load. With
        # `trunk_residual`, the trunk_dim -> trunk_dim layers between the first and the bottleneck
        # layer become residual blocks, so depth can be added without losing the input signal.
        widths = [fused_dim] + [trunk_dim] * (trunk_layers - 1) + [bottleneck_dim]
        trunk: list[nn.Module] = []
        for layer, (in_dim, out_dim) in enumerate(zip(widths[:-1], widths[1:])):
            if trunk_residual and 0 < layer < trunk_layers - 1:
                trunk.append(ResidualBlock(trunk_dim, trunk_dropout))
            else:
                trunk += [nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(trunk_dropout)]
        self.trunk = nn.Sequential(*trunk)
        self.classifier = nn.Linear(bottleneck_dim, n_terms)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = {}
        for group_name in self.group_names:
            group_features = features.index_select(1, getattr(self, f"{group_name}_indices"))
            group_projection = self.branches[group_name](group_features)
            if group_name == "promoter_agront":
                is_applicable = group_features.abs().sum(dim=1, keepdim=True) > 0
                group_projection = group_projection * is_applicable
            projected[group_name] = group_projection

        fused = [projected[group_name] for group_name in self.non_promoter_group_names]
        if self.promoter_fusion is not None:
            promoter_features = torch.cat(
                [projected[group_name] for group_name in self.promoter_group_names],
                dim=1,
            )
            fused.append(self.promoter_fusion(promoter_features))
        return self.classifier(self.trunk(torch.cat(fused, dim=1)))


class AsymmetricLoss(nn.Module):
    """Asymmetric focal loss for highly imbalanced multi-label targets."""

    def __init__(self, gamma_negative: float = 4.0, gamma_positive: float = 0.0, clip: float = 0.05) -> None:
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        positive_loss = targets * torch.log(probabilities.clamp_min(1e-8))
        if self.gamma_positive:
            positive_loss *= (1.0 - probabilities).pow(self.gamma_positive)

        negative_probabilities = 1.0 - probabilities
        if self.clip:
            negative_probabilities = (negative_probabilities + self.clip).clamp(max=1.0)
        negative_loss = (1.0 - targets) * torch.log(negative_probabilities.clamp_min(1e-8))
        if self.gamma_negative:
            negative_loss *= (1.0 - negative_probabilities).pow(self.gamma_negative)
        return -(positive_loss + negative_loss).mean()


def make_loss(
    loss_name: str,
    y_train: np.ndarray,
    max_pos_weight: float,
    gamma_negative: float,
    gamma_positive: float,
    asymmetric_clip: float,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray]:
    positive_counts = y_train.sum(axis=0, dtype=np.int64)
    negative_counts = len(y_train) - positive_counts
    pos_weights = np.ones(y_train.shape[1], dtype=np.float32)
    has_positives = positive_counts > 0
    pos_weights[has_positives] = np.sqrt(negative_counts[has_positives] / positive_counts[has_positives])
    pos_weights = np.clip(pos_weights, 1.0, max_pos_weight)

    if loss_name == "bce":
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    elif loss_name == "weighted_bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weights).to(device))
    elif loss_name == "asymmetric":
        criterion = AsymmetricLoss(
            gamma_negative=gamma_negative,
            gamma_positive=gamma_positive,
            clip=asymmetric_clip,
        )
    else:
        raise ValueError(f"Unknown loss {loss_name!r}")
    return criterion, pos_weights
