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
TargetKind = Literal["module", "leaf", "projection_segment"]
IdentityStability = Literal["model_owned", "path_derived"]
ProfileFidelity = Literal[
    "safe_default",
    "paper_exact",
    "reference_implementation",
    "model_family_recipe",
    "experimental",
]


@dataclass(frozen=True)
class TargetSpec:
    """Describe model leaves selected by paths, semantic tags, or predicates."""

    tags_all: tuple[str, ...] = ()
    tags_any: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    predicate: LeafPredicate | None = None
    min_depth: int | None = None
    max_depth: int | None = None
    target_kind: TargetKind = "leaf"
    allow_empty: bool = False


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
    method_name: str | None = None


@dataclass(frozen=True)
class MethodProfile:
    """Declared fidelity profile for a model-side fine-tuning method."""

    id: str
    method: str
    fidelity: ProfileFidelity
    reference_ids: tuple[str, ...]
    config: Mapping[str, Any]
    target_spec: Mapping[str, Any]
    known_deviations: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatePolicy:
    """Training-state semantics for stateful Equinox models."""

    training: Literal[
        "frozen",
        "microbatch_sequential",
        "optimizer_step_aggregate",
        "external",
    ] = "frozen"
    sam_second_pass: Literal["discard", "reuse_first_pass", "update"] = "discard"
    averaging: Literal[
        "do_not_average",
        "average_arrays",
        "recalibrate",
    ] = "recalibrate"


@dataclass(frozen=True)
class WeightLayout:
    """Logical layout metadata for matrix or convolution weights."""

    physical: Literal["out_in", "in_out", "conv_oihw", "conv_hwio"] = "out_in"
    input_axis: int = 1
    output_axis: int = 0
    groups: int = 1


@dataclass(frozen=True)
class ProjectionSegment:
    """Logical segment inside a fused projection weight."""

    name: Literal["q", "k", "v", "o", "gate", "up", "down", "custom"]
    axis: int
    start: int
    stop: int


@dataclass(frozen=True)
class ParamIdentity:
    """Stable logical identity for an optimizable or mergeable parameter."""

    logical_id: str
    module_id: str
    leaf_role: str
    physical_path: Path
    tags: frozenset[str]
    depth: int | None
    alias_group: str | None
    layout: WeightLayout | None
    segment: ProjectionSegment | None


@dataclass(frozen=True)
class FeatureSpec:
    """Reproducible feature endpoint and preprocessing metadata."""

    endpoint: str
    output_layout: Literal["BNC", "BCHW", "BTC", "BCT", "BC"]
    token_selection: Literal[
        "all",
        "cls",
        "patches",
        "frames",
        "last_valid",
        "custom",
    ]
    pooling: str | None
    mask_field: str | None = None
    exclude_prompt_tokens: bool = True
    normalize: Literal["none", "l2", "standardize"] = "none"
    layer_aggregation: Mapping[str, Any] | None = None
    preprocessing_fingerprint: str | None = None


@dataclass(frozen=True)
class CalibrationArtifact:
    """Immutable statistics consumed by data-aware fine-tuning methods."""

    kind: Literal[
        "activation_covariance",
        "activation_svd",
        "input_covariance",
        "quantization_residual",
        "fisher_diagonal",
    ]
    base_checkpoint_hash: str
    logical_parameter_ids: tuple[str, ...]
    statistics: PyTree
    sample_count: int
    data_fingerprint: str
    accumulation_dtype: str
    distributed_reduction: str


@dataclass(frozen=True)
class AuxLossSpec:
    """Optimizer-neutral auxiliary-loss declaration."""

    name: str
    registry_key: str
    coefficient_hint: float | None
    reduction: Literal["sum", "mean", "none"]
    normalizer: Literal["none", "parameters", "examples", "tokens"]
    required_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelLineage:
    """Lineage binding for plans, deltas, merges, and checkpoints."""

    base_model_name: str | None = None
    architecture_hash: str | None = None
    base_checkpoint_id: str | None = None
    base_checkpoint_hash: str | None = None
    base_value_hash: str | None = None
    preprocessing_fingerprint: str | None = None
    model_state_hash: str | None = None
    logical_id_table_hash: str | None = None
    quantization_fingerprint: str | None = None
    sharding_fingerprint: str | None = None
    model_revision: str | None = None
    parent_bundle_ids: tuple[str, ...] = ()
    identity_stability: IdentityStability = "path_derived"
    parent_lineages: tuple["ModelLineage", ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompactLeafMap:
    """Reversible compact-leaf mapping keyed by logical parameter ID."""

    logical_ids: tuple[str, ...]
    physical_paths: tuple[Path, ...]
    treedef_fingerprint: str


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
class FineTuneRecipe:
    """Declarative fine-tuning preset metadata.

    Recipes describe model-side changes and planning metadata. Optimizers,
    schedules, dataloaders, and training loops stay external.
    """

    name: str
    method: str
    head: HeadSpec | None
    peft: Any | None
    trainable: TrainableSpec
    labels: "LLRDConfig | None"
    notes: tuple[str, ...] = ()
    external_hints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SAMMetadata:
    """Metadata marker for external SAM/ASAM optimizer wrappers."""

    external_only: bool = True
    rho_hint: float = 0.05


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

    @classmethod
    def vit_base(cls, *, decay: float = 0.65) -> "LLRDConfig":
        """Return the ViT-B-style LLRD preset."""

        return cls(decay=decay)

    @classmethod
    def vit_large_or_huge(cls, *, decay: float = 0.75) -> "LLRDConfig":
        """Return the ViT-L/H-style LLRD preset."""

        return cls(decay=decay)

    @classmethod
    def audio_transformer(cls, *, decay: float = 0.75) -> "LLRDConfig":
        """Return the audio-transformer LLRD preset."""

        return cls(decay=decay)


@dataclass(frozen=True)
class ParamInfo:
    """Per-leaf metadata produced by later planning phases."""

    path: Path = ()
    logical_id: str = ""
    tags: frozenset[str] = frozenset()
    role: str = ""
    depth: int | None = None
    is_array: bool = False
    is_inexact_array: bool = False
    trainable: bool = False
    weight_decay: bool = False
    lr_multiplier: float | None = None
    label: str | None = None
    layout: WeightLayout | None = None
    segment: ProjectionSegment | None = None


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
    identities: PyTree
    model_state: eqx.nn.State | None
    state_policy: StatePolicy
    feature_spec: FeatureSpec | None
    aux_losses: tuple[AuxLossSpec, ...]
    profile: MethodProfile | None
    lineage: ModelLineage
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
    model_state: Any = None
    lineage: ModelLineage = field(default_factory=ModelLineage)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FineTuneBundleError(ValueError):
    """Raised when a fine-tuning bundle is malformed or incompatible."""


__all__ = (
    "AuxLossSpec",
    "CalibrationArtifact",
    "CompactLeafMap",
    "DepthAxis",
    "FeatureSpec",
    "FineTuneBundle",
    "FineTuneBundleError",
    "FineTunePlan",
    "FineTuneRecipe",
    "GroupSpec",
    "HeadSpec",
    "LLRDConfig",
    "MethodProfile",
    "ModelLineage",
    "ParamIdentity",
    "ParamInfo",
    "ProjectionSegment",
    "SAMMetadata",
    "StatePolicy",
    "TargetSpec",
    "TargetKind",
    "TrainableMode",
    "TrainableReport",
    "TrainableSpec",
    "WeightLayout",
)
