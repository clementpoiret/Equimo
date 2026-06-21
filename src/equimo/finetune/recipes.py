"""Baseline fine-tuning recipes."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax

from ._typing import PyTree
from .config import FineTunePlan, LLRDConfig, TargetSpec, TrainableSpec
from .feature_extraction import LinearProbe, make_linear_probe
from .peft.adapters import (
    AdaptFormerConfig,
    AdapterConfig,
    apply_adapters,
    apply_adaptformer,
)
from .pooling import PoolName
from .surgery import prepare_finetune
from .tags import iter_param_infos


@dataclass(frozen=True)
class LPFTRecipe:
    """Linear-probe then fine-tune recipe metadata."""

    stage1: TrainableSpec = field(default_factory=lambda: TrainableSpec(mode="head"))
    stage2: TrainableSpec = field(default_factory=lambda: TrainableSpec(mode="full"))
    preserve_trained_head: bool = True
    stage2_labels: LLRDConfig = field(default_factory=LLRDConfig)
    external_ft_lr_scale_hint: tuple[float, float] = (0.2, 0.5)

    def stage1_plan(self, model: PyTree) -> FineTunePlan:
        """Prepare the linear-probe stage."""

        return prepare_finetune(model, trainable=self.stage1)

    def stage2_plan(self, model: PyTree) -> FineTunePlan:
        """Prepare the fine-tuning stage without replacing the trained head."""

        return prepare_finetune(
            model,
            trainable=self.stage2,
            labels=self.stage2_labels,
        )


def linear_probe(
    backbone: PyTree,
    *,
    in_features: int,
    out_features: int,
    key: jax.Array,
    pool: PoolName = "auto",
) -> LinearProbe:
    """Create a linear-probe model wrapper."""

    return make_linear_probe(
        backbone,
        in_features=in_features,
        out_features=out_features,
        key=key,
        pool=pool,
    )


def head_plus_norm(model: PyTree) -> FineTunePlan:
    """Prepare a head-plus-norm fine-tuning plan."""

    return prepare_finetune(
        model,
        trainable=TrainableSpec(mode="head_plus_norm"),
    )


def full_ft_llrd(
    model: PyTree,
    *,
    decay: float = 0.75,
    freeze_patch_embed: bool = True,
) -> FineTunePlan:
    """Prepare a full fine-tuning plan with LLRD labels."""

    return prepare_finetune(
        model,
        trainable=TrainableSpec(
            mode="full",
            freeze=_patch_freeze(freeze_patch_embed),
        ),
        labels=LLRDConfig(decay=decay),
    )


def partial_ft_last_k_blocks(
    model: PyTree,
    *,
    k: int | str = "one_third",
    decay: float = 0.75,
    train_head: bool = True,
    train_norm: bool = True,
    freeze_patch_embed: bool = True,
) -> FineTunePlan:
    """Prepare a partial fine-tuning plan over the last ``k`` blocks."""

    depths = sorted({info.depth for info in iter_param_infos(model) if info.depth is not None})
    if not depths:
        depth_range = None
    else:
        count = _resolve_last_k(k, len(depths))
        selected = depths[-count:]
        depth_range = (selected[0], selected[-1] + 1)

    return prepare_finetune(
        model,
        trainable=TrainableSpec(
            mode="partial",
            depth_range=depth_range,
            train_head=train_head,
            train_norm=train_norm,
            freeze=_patch_freeze(freeze_patch_embed),
        ),
        labels=LLRDConfig(decay=decay),
    )


def lpft(
    *,
    stage1: TrainableSpec | None = None,
    stage2: TrainableSpec | None = None,
    stage2_labels: LLRDConfig | None = None,
    preserve_trained_head: bool = True,
) -> LPFTRecipe:
    """Return LP-FT stage metadata."""

    return LPFTRecipe(
        stage1=TrainableSpec(mode="head") if stage1 is None else stage1,
        stage2=TrainableSpec(mode="full") if stage2 is None else stage2,
        preserve_trained_head=preserve_trained_head,
        stage2_labels=LLRDConfig() if stage2_labels is None else stage2_labels,
    )


def adapter_transformer(
    model: PyTree,
    *,
    key: jax.Array,
    bottleneck: int | None = None,
    placement: str = "after_mlp",
) -> PyTree:
    """Apply a baseline transformer adapter configuration."""

    return apply_adapters(
        model,
        AdapterConfig(bottleneck=bottleneck, placement=placement),
        key=key,
    )


def adapter_transformer_strong(
    model: PyTree,
    *,
    key: jax.Array,
    bottleneck: int = 64,
) -> PyTree:
    """Apply a stronger two-placement adapter configuration."""

    return apply_adapters(
        model,
        AdapterConfig(bottleneck=bottleneck, placement="both"),
        key=key,
    )


def adaptformer_transformer(
    model: PyTree,
    *,
    key: jax.Array,
    bottleneck: int = 64,
) -> PyTree:
    """Apply AdaptFormer-style adapters."""

    return apply_adaptformer(
        model,
        AdaptFormerConfig(bottleneck=bottleneck),
        key=key,
    )


def _patch_freeze(enabled: bool) -> TargetSpec | None:
    return TargetSpec(tags=("embedding.patch",)) if enabled else None


def _resolve_last_k(k: int | str, depth_count: int) -> int:
    if isinstance(k, int):
        if k < 1:
            raise ValueError("k must be >= 1.")
        return min(k, depth_count)
    if k == "one_third":
        return max(1, depth_count // 3)
    raise ValueError("k must be an integer or 'one_third'.")


__all__ = (
    "LPFTRecipe",
    "adapter_transformer",
    "adapter_transformer_strong",
    "adaptformer_transformer",
    "full_ft_llrd",
    "head_plus_norm",
    "linear_probe",
    "lpft",
    "partial_ft_last_k_blocks",
)
