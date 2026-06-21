"""Language fine-tuning recipe presets."""

from __future__ import annotations

import jax

from .._typing import PyTree
from ..config import FineTunePlan, TargetSpec, TrainableSpec
from ..heads import ProjectionHead
from ..peft.lora import LoRAConfig, apply_lora
from ..peft.prefix import PrefixConfig, PrefixTunedModel, apply_prefixes
from ..surgery import prepare_finetune


def lora_encoder(
    model: PyTree,
    *,
    key: jax.Array,
    rank: int = 8,
    alpha: float = 16.0,
) -> PyTree:
    """Apply LoRA to a text encoder."""

    return apply_lora(
        model,
        LoRAConfig(
            rank=rank,
            alpha=alpha,
            target=TargetSpec(tags_any=("attention.qkv", "attention.proj")),
        ),
        key=key,
    )


def prefix_encoder(
    model: PyTree,
    *,
    key: jax.Array,
    num_prefix_tokens: int = 16,
) -> PrefixTunedModel:
    """Apply prefix tuning metadata to a text encoder."""

    return apply_prefixes(
        model,
        PrefixConfig(num_prefix_tokens=num_prefix_tokens),
        key=key,
    )


def projection_head(
    in_features: int,
    out_dim: int,
    *,
    key: jax.Array,
) -> ProjectionHead:
    """Create a projection head for language embeddings."""

    return ProjectionHead(in_features, out_dim, key=key)


def locked_tower(model: PyTree) -> FineTunePlan:
    """Freeze all language tower leaves."""

    return prepare_finetune(model, trainable=TrainableSpec(mode="frozen"))


__all__ = (
    "locked_tower",
    "lora_encoder",
    "prefix_encoder",
    "projection_head",
)
