"""Configuration shells for Equimo fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import equinox as eqx

from ._typing import LeafPredicate, Path, PyTree

TrainableMode = Literal[
    "frozen",
    "head",
    "head_plus_norm",
    "norm",
    "bias",
    "scale_shift",
    "partial",
    "surgical",
    "full",
    "peft",
]

DepthAxis = Literal["block", "stage", "module"]


@dataclass(frozen=True)
class TargetSpec:
    """Describe model leaves selected by paths, semantic tags, or predicates."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    predicate: LeafPredicate | None = None
    min_depth: int | None = None
    max_depth: int | None = None


@dataclass(frozen=True)
class TrainableSpec:
    """Describe the intended trainability policy for a model."""

    mode: TrainableMode
    target: TargetSpec | None = None
    freeze: TargetSpec | None = None
    train_head: bool = True
    train_norm: bool = False
    train_bias: bool = False
    depth_range: tuple[int, int] | None = None
    shift: str | None = None
    method_name: str | None = None


@dataclass(frozen=True)
class HeadSpec:
    """Declarative task-head metadata for recipe presets."""

    kind: str = "linear"
    in_features: int | None = None
    out_features: int | None = None
    hidden_dim: int | None = None
    num_layers: int = 1
    bias: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLRDConfig:
    """Layer-wise learning-rate decay metadata."""

    decay: float = 0.75
    depth_axis: DepthAxis = "block"
    top_block_lr_multiplier: float = 1.0
    head_lr_multiplier: float = 1.0
    embedding_lr_multiplier: float | None = None
    rebase_selected_depth: bool = False
    no_weight_decay_tags: tuple[str, ...] = (
        "bias",
        "norm",
        "embedding.position",
        "embedding.class_token",
        "embedding.register_token",
        "embedding.distillation_token",
        "embedding.mask_token",
    )
    block_label_format: str = "block_{depth:02d}"
    head_label: str = "head"
    embedding_label: str = "embed"


@dataclass(frozen=True)
class ParamInfo:
    """Per-leaf metadata produced by later planning phases."""

    path: Path = ()
    tags: frozenset[str] = frozenset()
    role: str = ""
    depth: int | None = None
    is_array: bool = False
    is_inexact_array: bool = False
    trainable: bool = False
    weight_decay: bool = False
    lr_multiplier: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class GroupSpec:
    """Metadata for one optimizer group label."""

    label: str
    role: str
    depth: int | None
    lr_multiplier: float
    weight_decay: bool
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainableReport:
    """Summary of trainable and frozen parameter leaves."""

    total_params: int = 0
    trainable_params: int = 0
    trainable_fraction: float = 0.0
    trainable_by_label: Mapping[str, int] = field(default_factory=dict)
    frozen_by_label: Mapping[str, int] = field(default_factory=dict)
    adapter_params: int = 0
    head_params: int = 0
    mergeable: bool = False
    estimated_delta_size_bytes: int = 0
    target_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class FineTunePlan:
    """Partitioned model state and metadata prepared for external optimizers."""

    trainable: PyTree
    frozen: PyTree
    labels: PyTree
    group_specs: Mapping[str, GroupSpec]
    trainable_mask: PyTree
    param_info: PyTree
    report: TrainableReport

    def combine(self, trainable: PyTree | None = None) -> PyTree:
        """Combine trainable leaves with the stored frozen tree."""

        return eqx.combine(
            self.trainable if trainable is None else trainable, self.frozen
        )


@dataclass(frozen=True)
class FineTuneBundle:
    """Portable fine-tuning delta bundle schema shell."""

    method: str = ""
    schema_version: int = 1
    base_model_name: str | None = None
    base_model_config: Mapping[str, Any] = field(default_factory=dict)
    base_checkpoint_id: str | None = None
    equimo_version: str = ""
    architecture_hash: str = ""
    adapter_config: Mapping[str, Any] = field(default_factory=dict)
    selector_spec: Mapping[str, Any] = field(default_factory=dict)
    trainable_labels: Any = None
    delta_tree: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = (
    "DepthAxis",
    "FineTuneBundle",
    "FineTunePlan",
    "GroupSpec",
    "HeadSpec",
    "LLRDConfig",
    "ParamInfo",
    "TargetSpec",
    "TrainableMode",
    "TrainableReport",
    "TrainableSpec",
)
