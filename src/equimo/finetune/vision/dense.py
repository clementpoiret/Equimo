"""Dense vision fine-tuning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax

from ..heads import ActivationName, DenseFeatureAdapter
from ..regularization import FeatureDistillationConfig


@dataclass(frozen=True)
class DenseVisionConfig:
    """Configuration for dense vision transfer helpers."""

    activation: ActivationName = "identity"
    dropout: float = 0.0
    bias: bool = True
    distillation: FeatureDistillationConfig = field(
        default_factory=FeatureDistillationConfig.dense
    )


def dense_feature_adapter(
    in_features: int,
    out_features: int,
    *,
    key: jax.Array,
    config: DenseVisionConfig | None = None,
) -> DenseFeatureAdapter:
    """Create a dense feature adapter for spatial or token features."""

    config = DenseVisionConfig() if config is None else config
    return DenseFeatureAdapter(
        in_features,
        out_features,
        key=key,
        activation=config.activation,
        dropout=config.dropout,
        bias=config.bias,
    )


def dense_distillation_config(
    *,
    layers: tuple[str, ...] = ("25%", "50%", "75%", "100%"),
    normalize_features: bool = True,
) -> FeatureDistillationConfig:
    """Return the dense-task feature distillation preset."""

    return FeatureDistillationConfig.dense(
        layers=layers,
        normalize_features=normalize_features,
    )


__all__ = (
    "DenseVisionConfig",
    "dense_distillation_config",
    "dense_feature_adapter",
)
