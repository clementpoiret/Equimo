"""Continued self-supervised adaptation planning helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import jax

from ._typing import Path, PyTree
from .config import FineTunePlan, TargetSpec, TrainableSpec
from .peft.lora import LoRAConfig, apply_lora, lora_config_to_dict
from .surgery import prepare_finetune
from .tags import canonical_tags_for_path, infer_depth, iter_param_infos


UnfreezePolicy = Literal["first", "last"]


@dataclass(frozen=True)
class ContinuedSSLAdaptationConfig:
    """Configuration for target-domain continued SSL adaptation."""

    peft: LoRAConfig | None = field(
        default_factory=lambda: LoRAConfig(rank=8, alpha=16)
    )
    unfreeze_blocks: int = 1
    unfreeze_policy: UnfreezePolicy = "last"
    train_norm: bool = True
    train_head: bool = False
    save_stage: str = "continued_ssl_delta"


@dataclass(frozen=True)
class SupervisedAfterSSLConfig:
    """Configuration for supervised fine-tuning after continued SSL."""

    peft: LoRAConfig | None = field(
        default_factory=lambda: LoRAConfig(rank=8, alpha=16)
    )
    train_head: bool = True
    reuse_ssl_delta: bool = True


@dataclass(frozen=True)
class ContinuedSSLPlan:
    """Fine-tuning plan plus continued-SSL serialization metadata."""

    plan: FineTunePlan
    stage: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def combine(self, trainable: PyTree | None = None) -> PyTree:
        """Delegate to the wrapped ``FineTunePlan``."""

        return self.plan.combine(trainable)


def continued_ssl_trainable_spec(
    model: PyTree,
    config: ContinuedSSLAdaptationConfig | None = None,
) -> TrainableSpec:
    """Build the trainability mask for continued SSL adaptation."""

    config = ContinuedSSLAdaptationConfig() if config is None else config
    if config.peft is None and config.unfreeze_blocks <= 0 and not config.train_norm:
        return TrainableSpec(mode="head" if config.train_head else "frozen")

    active_depths = _selected_depths(model, config.unfreeze_blocks, config.unfreeze_policy)
    peft_tag = _peft_tag(config.peft)

    def predicate(path: Path, leaf: Any) -> bool:
        tags = canonical_tags_for_path(path, leaf)
        if peft_tag is not None and peft_tag in tags:
            return True
        if config.train_norm and "norm" in tags:
            return True
        depth = infer_depth(path)
        return depth in active_depths and "block" in tags

    return TrainableSpec(
        mode="surgical",
        target=TargetSpec(predicate=predicate),
        train_head=config.train_head,
    )


def supervised_after_ssl_trainable_spec(
    config: SupervisedAfterSSLConfig | None = None,
) -> TrainableSpec:
    """Build the supervised follow-up trainability mask."""

    config = SupervisedAfterSSLConfig() if config is None else config
    method = _peft_method(config.peft)
    if method is None:
        return TrainableSpec(mode="head" if config.train_head else "frozen")
    return TrainableSpec(
        mode="peft",
        method_name=method,
        train_head=config.train_head,
    )


def continued_ssl_adaptation(
    model: PyTree,
    config: ContinuedSSLAdaptationConfig | None = None,
    *,
    key: jax.Array,
) -> ContinuedSSLPlan:
    """Apply PEFT wrappers and prepare a continued-SSL trainability plan."""

    config = ContinuedSSLAdaptationConfig() if config is None else config
    adapted = _apply_peft(model, config.peft, key=key)
    plan = prepare_finetune(
        adapted,
        trainable=continued_ssl_trainable_spec(adapted, config),
    )
    return ContinuedSSLPlan(
        plan=plan,
        stage=config.save_stage,
        metadata=_continued_ssl_metadata(config),
    )


def supervised_after_ssl(
    model: PyTree,
    config: SupervisedAfterSSLConfig | None = None,
    *,
    key: jax.Array | None = None,
) -> ContinuedSSLPlan:
    """Prepare the supervised stage after a continued-SSL delta."""

    config = SupervisedAfterSSLConfig() if config is None else config
    if config.reuse_ssl_delta:
        adapted = model
    else:
        if key is None:
            raise ValueError("A PRNG key is required when reuse_ssl_delta=False.")
        adapted = _apply_peft(model, config.peft, key=key)
    plan = prepare_finetune(
        adapted,
        trainable=supervised_after_ssl_trainable_spec(config),
    )
    return ContinuedSSLPlan(
        plan=plan,
        stage="supervised_after_ssl",
        metadata=_supervised_after_ssl_metadata(config),
    )


def _apply_peft(model: PyTree, peft: LoRAConfig | None, *, key: jax.Array) -> PyTree:
    if peft is None:
        return model
    if isinstance(peft, LoRAConfig):
        return apply_lora(model, peft, key=key)
    raise ValueError(f"Unsupported continued SSL PEFT config {type(peft).__name__}.")


def _selected_depths(
    model: PyTree,
    count: int,
    policy: UnfreezePolicy,
) -> frozenset[int]:
    if count <= 0:
        return frozenset()
    depths = sorted({info.depth for info in iter_param_infos(model) if info.depth is not None})
    if not depths:
        return frozenset()
    count = min(count, len(depths))
    if policy == "last":
        return frozenset(depths[-count:])
    if policy == "first":
        return frozenset(depths[:count])
    raise ValueError(f"Unsupported unfreeze_policy {policy!r}.")


def _peft_method(peft: LoRAConfig | None) -> str | None:
    if peft is None:
        return None
    if isinstance(peft, LoRAConfig):
        return "lora"
    raise ValueError(f"Unsupported PEFT config {type(peft).__name__}.")


def _peft_tag(peft: LoRAConfig | None) -> str | None:
    return _peft_method(peft)


def _peft_metadata(peft: LoRAConfig | None) -> Mapping[str, Any] | None:
    if peft is None:
        return None
    if isinstance(peft, LoRAConfig):
        return {"method": "lora", "config": lora_config_to_dict(peft)}
    raise ValueError(f"Unsupported PEFT config {type(peft).__name__}.")


def _continued_ssl_metadata(config: ContinuedSSLAdaptationConfig) -> Mapping[str, Any]:
    return {
        "stage": config.save_stage,
        "peft": _peft_metadata(config.peft),
        "unfreeze_blocks": config.unfreeze_blocks,
        "unfreeze_policy": config.unfreeze_policy,
        "train_norm": config.train_norm,
        "train_head": config.train_head,
    }


def _supervised_after_ssl_metadata(config: SupervisedAfterSSLConfig) -> Mapping[str, Any]:
    return {
        "stage": "supervised_after_ssl",
        "peft": _peft_metadata(config.peft),
        "train_head": config.train_head,
        "reuse_ssl_delta": config.reuse_ssl_delta,
    }


__all__ = (
    "ContinuedSSLAdaptationConfig",
    "ContinuedSSLPlan",
    "SupervisedAfterSSLConfig",
    "continued_ssl_adaptation",
    "continued_ssl_trainable_spec",
    "supervised_after_ssl",
    "supervised_after_ssl_trainable_spec",
)
