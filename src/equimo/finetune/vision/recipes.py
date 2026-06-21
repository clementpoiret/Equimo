"""Vision fine-tuning recipe presets."""

from __future__ import annotations

import jax

from .._typing import PyTree
from ..config import FineTunePlan, TargetSpec
from ..feature_extraction import LinearProbe
from ..peft.lora import LoRAConfig, apply_lora
from ..peft.prompts import PromptedModel, VPTDeepConfig, VPTShallowConfig, apply_prompts
from ..recipes import (
    adapter_transformer,
    adaptformer_transformer,
    full_ft_llrd,
    head_plus_norm,
    linear_probe,
    partial_ft_last_k_blocks,
)


def linear_probe_vit(
    model: PyTree,
    *,
    in_features: int,
    out_features: int,
    key: jax.Array,
    pool: str = "cls",
) -> LinearProbe:
    """Build a ViT linear-probe wrapper."""

    return linear_probe(
        model,
        in_features=in_features,
        out_features=out_features,
        key=key,
        pool=pool,
    )


def head_plus_norm_vit(model: PyTree) -> FineTunePlan:
    """Prepare a ViT head-plus-norm plan."""

    return head_plus_norm(model)


def full_ft_vit_llrd(
    model: PyTree,
    *,
    decay: float = 0.75,
    freeze_patch_embed: bool = True,
) -> FineTunePlan:
    """Prepare a ViT full fine-tuning plan with LLRD."""

    return full_ft_llrd(
        model,
        decay=decay,
        freeze_patch_embed=freeze_patch_embed,
    )


def partial_ft_vit_llrd(
    model: PyTree,
    *,
    last_k_blocks: int | str = "one_third",
    decay: float = 0.75,
) -> FineTunePlan:
    """Prepare a ViT partial fine-tuning plan over the last blocks."""

    return partial_ft_last_k_blocks(
        model,
        k=last_k_blocks,
        decay=decay,
    )


def lora_vit(
    model: PyTree,
    *,
    key: jax.Array,
    rank: int = 8,
    alpha: float = 16.0,
    target: tuple[str, ...] = ("attention.qkv", "attention.proj"),
) -> PyTree:
    """Apply LoRA to a ViT-like model."""

    return apply_lora(
        model,
        LoRAConfig(rank=rank, alpha=alpha, target=TargetSpec(tags_any=target)),
        key=key,
    )


def adapter_vit(
    model: PyTree,
    *,
    key: jax.Array,
    bottleneck: int = 64,
    placement: str = "after_mlp",
) -> PyTree:
    """Apply bottleneck adapters to a ViT-like model."""

    return adapter_transformer(
        model,
        key=key,
        bottleneck=bottleneck,
        placement=placement,
    )


def adaptformer_vit(
    model: PyTree,
    *,
    key: jax.Array,
    bottleneck: int = 64,
) -> PyTree:
    """Apply AdaptFormer-style adapters to a ViT-like model."""

    return adaptformer_transformer(model, key=key, bottleneck=bottleneck)


def vpt_vit(
    model: PyTree,
    *,
    key: jax.Array,
    num_tokens: int = 10,
    depth: str = "deep",
) -> PromptedModel:
    """Apply visual prompt tuning to a ViT-like model."""

    prompt_config = (
        VPTDeepConfig(num_tokens=num_tokens)
        if depth == "deep"
        else VPTShallowConfig(num_tokens=num_tokens)
    )
    return apply_prompts(
        model,
        prompt_config,
        key=key,
    )


__all__ = (
    "adapter_vit",
    "adaptformer_vit",
    "full_ft_vit_llrd",
    "head_plus_norm_vit",
    "linear_probe_vit",
    "lora_vit",
    "partial_ft_vit_llrd",
    "vpt_vit",
)
