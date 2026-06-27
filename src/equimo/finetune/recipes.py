"""Baseline fine-tuning recipes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import equinox as eqx
import jax

from ._typing import Path, PyTree
from .config import FineTunePlan, LLRDConfig, TargetSpec, TrainableSpec
from .feature_extraction import LinearProbe, make_linear_probe
from .peft.adapters import (
    AdaptFormerConfig,
    AdapterConfig,
    add_adapter,
    apply_adapters,
    apply_adaptformer,
)
from .peft.lora import LoRAConfig, apply_lora
from .peft.prompts import PromptedModel, VPTDeepConfig, apply_prompts
from .pooling import PoolName
from .selectors import is_linear
from .surgery import prepare_finetune
from .tags import Tagger, canonical_tags_for_path, infer_depth, iter_param_infos


@dataclass(frozen=True)
class LinearProbeConfig:
    """Configuration metadata for linear probing."""

    pool: str = "auto"
    feature_norm: str | None = None
    head: str = "linear"
    head_bias: bool = True
    head_init: str = "trunc_normal_0.02"
    train_backbone: bool = False
    train_head: bool = True
    cache_features: bool = False


@dataclass(frozen=True)
class LinearProbeRecipe:
    """Recipe metadata for linear probing."""

    pool: str = "auto"
    feature_norm: str = "l2_or_standardize"
    head: str = "linear"


@dataclass(frozen=True)
class HeadPlusNormConfig:
    """Configuration metadata for head-plus-norm tuning."""

    train_head: bool = True
    train_norm: bool = True
    norm_types: tuple[str, ...] = ("LayerNorm", "RMSNorm", "BatchNorm", "GroupNorm")
    train_norm_scale: bool = True
    train_norm_bias: bool = True
    train_bias: bool = False
    bn_stats_policy: str = "frozen"
    include_embeddings: bool = False
    include_positional_parameters: bool = False

    def trainable_spec(
        self,
        *,
        tagger: Tagger = canonical_tags_for_path,
    ) -> TrainableSpec:
        """Return a trainability mask matching this head-plus-norm preset."""

        if not any(
            (
                self.train_head,
                self.train_norm,
                self.train_bias,
                self.include_embeddings,
                self.include_positional_parameters,
            )
        ):
            return TrainableSpec(mode="frozen")

        def predicate(path: Path, leaf: Any) -> bool:
            if not eqx.is_inexact_array(leaf):
                return False
            tags = frozenset(tagger(path, leaf))
            leaf_name = str(path[-1]) if path else ""
            if self.train_head and "head" in tags:
                return True
            if self.train_norm and "norm" in tags:
                if leaf_name == "bias":
                    return self.train_norm_bias
                if leaf_name in {"weight", "scale"}:
                    return self.train_norm_scale
                return True
            if self.train_bias and "bias" in tags:
                return True
            if self.include_positional_parameters and "embedding.position" in tags:
                return True
            if self.include_embeddings:
                return any(
                    tag.startswith("embedding.") and tag != "embedding.position"
                    for tag in tags
                )
            return False

        return TrainableSpec(
            mode="surgical",
            target=TargetSpec(predicate=predicate),
            train_head=False,
            train_norm=False,
            train_bias=False,
        )


@dataclass(frozen=True)
class PartialUnfreezeConfig:
    """Configuration metadata for partial unfreezing."""

    span: Literal["last"] = "last"
    fraction: float = 1 / 3
    min_blocks: int = 1
    train_head: bool = True
    train_norm: bool = True
    train_embeddings: bool = False
    train_positional_parameters: bool = False


@dataclass(frozen=True)
class LPFTRecipe:
    """Linear-probe then fine-tune recipe metadata."""

    stage1: TrainableSpec = field(default_factory=lambda: TrainableSpec(mode="head"))
    stage2: TrainableSpec = field(default_factory=lambda: TrainableSpec(mode="full"))
    preserve_trained_head: bool = True
    stage2_labels: LLRDConfig = field(default_factory=LLRDConfig)
    external_ft_lr_scale_hint: tuple[float, float] = (0.2, 0.5)

    def stage1_plan(
        self,
        model: PyTree,
        *,
        tagger: Tagger = canonical_tags_for_path,
    ) -> FineTunePlan:
        """Prepare the linear-probe stage."""

        return prepare_finetune(model, trainable=self.stage1, tagger=tagger)

    def stage2_plan(
        self,
        model: PyTree,
        *,
        tagger: Tagger = canonical_tags_for_path,
    ) -> FineTunePlan:
        """Prepare the fine-tuning stage without replacing the trained head."""

        if not self.preserve_trained_head:
            raise ValueError(
                "LPFTRecipe.stage2_plan currently requires preserve_trained_head=True; "
                "reset or replace the head explicitly before building the stage-2 plan."
            )
        return prepare_finetune(
            model,
            trainable=self.stage2,
            labels=self.stage2_labels,
            tagger=tagger,
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


def head_plus_norm(
    model: PyTree,
    config: HeadPlusNormConfig | None = None,
    *,
    labels: LLRDConfig | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> FineTunePlan:
    """Prepare a head-plus-norm fine-tuning plan."""

    config = HeadPlusNormConfig() if config is None else config
    return prepare_finetune(
        model,
        trainable=config.trainable_spec(tagger=tagger),
        labels=labels,
        tagger=tagger,
    )


def full_ft_llrd(
    model: PyTree,
    *,
    decay: float = 0.75,
    freeze_patch_embed: bool = True,
    labels: LLRDConfig | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> FineTunePlan:
    """Prepare a full fine-tuning plan with LLRD labels."""

    return prepare_finetune(
        model,
        trainable=TrainableSpec(
            mode="full",
            freeze=_patch_freeze(freeze_patch_embed),
        ),
        labels=LLRDConfig(decay=decay) if labels is None else labels,
        tagger=tagger,
    )


def partial_ft_last_k_blocks(
    model: PyTree,
    *,
    k: int | str = "one_third",
    decay: float = 0.75,
    train_head: bool = True,
    train_norm: bool = True,
    freeze_patch_embed: bool = True,
    labels: LLRDConfig | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> FineTunePlan:
    """Prepare a partial fine-tuning plan over the last ``k`` blocks."""

    depths = sorted(
        {
            info.depth
            for info in iter_param_infos(model, tagger=tagger)
            if info.depth is not None
        }
    )
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
        labels=LLRDConfig(decay=decay) if labels is None else labels,
        tagger=tagger,
    )


def partial_unfreeze(
    model: PyTree,
    config: PartialUnfreezeConfig | None = None,
    *,
    labels: LLRDConfig | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> FineTunePlan:
    """Prepare a partial-unfreeze plan from ``PartialUnfreezeConfig``."""

    config = PartialUnfreezeConfig() if config is None else config
    if config.span != "last":
        raise ValueError("PartialUnfreezeConfig currently supports span='last'.")
    depths = sorted(
        {
            info.depth
            for info in iter_param_infos(model, tagger=tagger)
            if info.depth is not None
        }
    )
    selected_depths = frozenset(
        _last_fraction_depths(depths, config.fraction, config.min_blocks)
    )

    def predicate(path: Path, leaf: Any) -> bool:
        tags = frozenset(tagger(path, leaf))
        depth = _path_depth(path)
        if depth in selected_depths:
            return True
        if config.train_embeddings and "embedding.patch" in tags:
            return True
        return config.train_positional_parameters and "embedding.position" in tags

    return prepare_finetune(
        model,
        trainable=TrainableSpec(
            mode="surgical",
            target=TargetSpec(predicate=predicate),
            train_head=config.train_head,
            train_norm=config.train_norm,
        ),
        labels=LLRDConfig() if labels is None else labels,
        tagger=tagger,
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


def lora_transformer(
    model: PyTree,
    *,
    key: jax.Array,
    rank: int = 8,
    alpha: float = 16.0,
    target: tuple[str, ...] = ("attention.qkv", "attention.proj"),
) -> PyTree:
    """Apply the default transformer LoRA recipe."""

    return apply_lora(
        model,
        LoRAConfig(rank=rank, alpha=alpha, target=TargetSpec(tags_any=target)),
        key=key,
    )


def lora_transformer_all_linear(
    model: PyTree,
    *,
    key: jax.Array,
    rank: int = 8,
    alpha: float = 16.0,
) -> PyTree:
    """Apply LoRA to every linear module selected by the generic predicate."""

    return apply_lora(
        model,
        LoRAConfig(rank=rank, alpha=alpha, target=TargetSpec(predicate=is_linear)),
        key=key,
    )


def vpt_deep(
    model: PyTree,
    *,
    key: jax.Array,
    num_tokens: int = 10,
) -> PromptedModel:
    """Apply the default deep visual prompt-tuning recipe."""

    return apply_prompts(
        model,
        VPTDeepConfig(num_tokens=num_tokens),
        key=key,
    )


def task_adapter_bank(
    model: PyTree,
    *,
    key: jax.Array,
    names: tuple[str, ...] = ("default",),
    bottleneck: int = 64,
    placement: str = "after_mlp",
) -> PyTree:
    """Add a named adapter bank for one or more tasks."""

    if not names:
        raise ValueError("task_adapter_bank requires at least one adapter name.")
    updated = model
    for name, adapter_key in zip(names, jax.random.split(key, len(names)), strict=True):
        updated = add_adapter(
            updated,
            name=name,
            config=AdapterConfig(bottleneck=bottleneck, placement=placement),
            key=adapter_key,
        )
    return updated


def _patch_freeze(enabled: bool) -> TargetSpec | None:
    return TargetSpec(tags_any=("embedding.patch",)) if enabled else None


def _resolve_last_k(k: int | str, depth_count: int) -> int:
    if isinstance(k, int):
        if k < 1:
            raise ValueError("k must be >= 1.")
        return min(k, depth_count)
    if k == "one_third":
        return max(1, depth_count // 3)
    raise ValueError("k must be an integer or 'one_third'.")


def _last_fraction_depths(
    depths: list[int],
    fraction: float,
    min_blocks: int,
) -> tuple[int, ...]:
    if not depths:
        return ()
    count = _span_count(len(depths), fraction, min_blocks)
    return tuple(depths[-count:])


def _span_count(depth_count: int, fraction: float, min_blocks: int) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("span fraction must satisfy 0 < fraction <= 1.")
    if min_blocks < 1:
        raise ValueError("min_blocks must be >= 1.")
    return min(depth_count, max(min_blocks, math.ceil(depth_count * fraction)))


def _path_depth(path: Path) -> int | None:
    return infer_depth(path)


__all__ = (
    "LPFTRecipe",
    "HeadPlusNormConfig",
    "LinearProbeConfig",
    "LinearProbeRecipe",
    "PartialUnfreezeConfig",
    "adapter_transformer",
    "adapter_transformer_strong",
    "adaptformer_transformer",
    "full_ft_llrd",
    "head_plus_norm",
    "linear_probe",
    "lora_transformer",
    "lora_transformer_all_linear",
    "lpft",
    "partial_ft_last_k_blocks",
    "partial_unfreeze",
    "task_adapter_bank",
    "vpt_deep",
)
