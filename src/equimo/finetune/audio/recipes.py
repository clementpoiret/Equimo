"""Audio fine-tuning recipe presets."""

from __future__ import annotations

import jax

from .._typing import PyTree
from ..feature_extraction import LinearProbe
from ..heads import CTCHead, MultiLabelHead
from ..config import TargetSpec
from ..peft.lora import LoRAConfig, apply_lora
from ..peft.adapters import AdapterConfig, apply_adapters
from ..pooling import MeanFramePool
from ..recipes import linear_probe


def linear_probe_ast(
    model: PyTree,
    *,
    in_features: int,
    out_features: int,
    key: jax.Array,
    pool: str = "mean_frame",
) -> LinearProbe:
    """Build an AST-style linear-probe wrapper."""

    return linear_probe(
        model,
        in_features=in_features,
        out_features=out_features,
        key=key,
        pool=pool,
    )


def multilabel_tagging_head(
    in_features: int,
    out_features: int,
    *,
    key: jax.Array,
) -> MultiLabelHead:
    """Create a raw-logit audio tagging head."""

    return MultiLabelHead(in_features, out_features, key=key)


def mean_frame_pool() -> MeanFramePool:
    """Create a frame-mean pooling module."""

    return MeanFramePool()


def ctc_head(
    in_features: int,
    vocab_size: int,
    *,
    key: jax.Array,
    blank_id: int = 0,
) -> CTCHead:
    """Create a raw-logit CTC head."""

    return CTCHead(in_features, vocab_size, key=key, blank_id=blank_id)


def adapter_ast(
    model: PyTree,
    *,
    key: jax.Array,
    bottleneck: int = 64,
    placement: str = "after_mlp",
) -> PyTree:
    """Apply transformer adapters to an AST-like model."""

    return apply_adapters(
        model,
        AdapterConfig(bottleneck=bottleneck, placement=placement),
        key=key,
    )


def lora_ast(
    model: PyTree,
    *,
    key: jax.Array,
    rank: int = 8,
    alpha: float = 16.0,
    target: tuple[str, ...] = ("attention.qkv", "attention.proj"),
) -> PyTree:
    """Apply LoRA to an AST-like model."""

    return apply_lora(
        model,
        LoRAConfig(rank=rank, alpha=alpha, target=TargetSpec(tags=target)),
        key=key,
    )


__all__ = (
    "ctc_head",
    "adapter_ast",
    "linear_probe_ast",
    "lora_ast",
    "mean_frame_pool",
    "multilabel_tagging_head",
)
