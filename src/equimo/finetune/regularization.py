"""Pure regularization helpers for fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import AuxLossSpec
from .paths import key_path_to_path, path_to_str
from .tags import canonical_tags_for_path


@dataclass(frozen=True)
class L2SPConfig:
    """L2-SP penalty metadata."""

    shared_mask: str = "transferred_trainable"
    new_mask: str = "newly_initialized"
    alpha: float = 1e-3
    beta: float = 0.0
    reduction: Literal["sum", "mean_per_parameter"] = "sum"
    anchor: str = "pretrained"
    library_variant: str = "paper_objective"
    coefficient_hint: float | None = None
    sweep_hint: tuple[float, float] = (1e-5, 1e-1)


@dataclass(frozen=True)
class FeatureDistillationConfig:
    """Generic teacher-feature distillation metadata.

    This is not a DELTA attention-transfer configuration.
    """

    layers: tuple[str, ...] = ("50%", "100%")
    metric: str = "mse"
    normalize_features: bool = True
    stop_gradient_teacher: bool = True
    reduction: Literal["mean", "sum"] = "mean"
    coefficient_hint: float = 1.0

    @classmethod
    def dense(
        cls,
        *,
        layers: tuple[str, ...] = ("25%", "50%", "75%", "100%"),
        normalize_features: bool = True,
    ) -> "FeatureDistillationConfig":
        """Return a dense-task feature distillation preset."""

        return cls(layers=layers, normalize_features=normalize_features)


@dataclass(frozen=True)
class DELTAAttentionConfig:
    """DELTA-style attention-weighted feature-map transfer metadata."""

    layers: tuple[str, ...] = ("100%",)
    attention: str = "learned_feature_map_attention"
    feature_alignment: Mapping[str, Any] = field(default_factory=dict)
    teacher: str = "pretrained_anchor"
    stop_gradient_teacher: bool = True
    reduction: Literal["sum", "mean"] = "sum"


@dataclass(frozen=True)
class EWCConfig:
    """Elastic Weight Consolidation penalty metadata."""

    fisher: Literal["diagonal"] = "diagonal"
    reduction: Literal["sum", "mean"] = "sum"
    coefficient_hint: float | None = None
    anchor: str = "previous_task"


@dataclass(frozen=True)
class MixoutConfig:
    """Mixout metadata for anchored stochastic parameter substitution."""

    p: float = 0.1
    anchor: str = "pretrained"
    target: str = "full_or_partial_trainable_weights"


def l2_sp_loss(
    model: PyTree,
    reference: PyTree,
    *,
    config: L2SPConfig | None = None,
    coefficient: float | None = None,
    reduction: Literal["sum", "mean_per_parameter"] | None = None,
) -> jax.Array:
    """Return an optionally scaled L2-SP penalty between compatible leaves."""

    resolved_config = L2SPConfig() if config is None else config
    resolved_reduction = resolved_config.reduction if reduction is None else reduction
    entries = _matched_l2_entries(
        model,
        reference,
        resolved_config,
    )
    shared_pairs = tuple(
        _l2_leaf(leaf, ref)
        for path, leaf, ref in entries
        if _include_l2_path(path, leaf, resolved_config)
    )
    new_pairs = tuple(
        _norm_leaf(leaf)
        for path, leaf, _ in entries
        if _include_l2_new_path(path, leaf, resolved_config)
    )
    if not shared_pairs and not new_pairs:
        return jnp.asarray(0.0)
    shared_penalty = _reduce_l2sp_terms_from_pairs(shared_pairs, resolved_reduction)
    new_penalty = _reduce_l2sp_terms_from_pairs(new_pairs, resolved_reduction)
    if coefficient is not None:
        return (shared_penalty + new_penalty) * coefficient
    return shared_penalty * (resolved_config.alpha / 2.0) + new_penalty * (
        resolved_config.beta / 2.0
    )


def adapter_norm_loss(
    model: PyTree,
    *,
    tags: tuple[str, ...] = (
        "adapter",
        "lora",
        "dora",
        "prompt",
        "prefix",
        "ia3",
        "scale_shift",
        "vera",
    ),
    reduction: Literal["mean", "sum"] = "mean",
    coefficient: float | None = None,
) -> jax.Array:
    """Return an optionally scaled norm penalty over adapter/PEFT leaves."""

    tag_set = frozenset(tags)
    pairs = tuple(
        _norm_leaf(leaf)
        for path, leaf in _inexact_array_entries(model)
        if not canonical_tags_for_path(path, leaf).isdisjoint(tag_set)
    )
    if not pairs:
        return jnp.asarray(0.0)
    sums, counts = zip(*pairs, strict=True)
    penalty = _reduce_terms(sums, counts, reduction)
    return penalty if coefficient is None else penalty * coefficient


def adalora_orthogonality_loss(
    model: PyTree,
    *,
    coefficient: float | None = None,
) -> jax.Array:
    """Return AdaLoRA triplet orthogonality loss over all adapters."""

    from .peft.lora import AdaLoRAModule

    losses = tuple(
        leaf.orthogonality_loss()
        for leaf in jtu.tree_leaves(
            model, is_leaf=lambda x: isinstance(x, AdaLoRAModule)
        )
        if isinstance(leaf, AdaLoRAModule)
    )
    if not losses:
        return jnp.asarray(0.0)
    loss = sum(losses, jnp.asarray(0.0, dtype=jnp.float32))
    return loss if coefficient is None else loss * coefficient


def adalora_orthogonality_aux_loss_spec(
    *,
    coefficient_hint: float = 0.5,
) -> AuxLossSpec:
    """Return the explicit AuxLossSpec for AdaLoRA orthogonality regularization."""

    return AuxLossSpec(
        name="adalora_orthogonality",
        registry_key="equimo.adalora_orthogonality_loss",
        coefficient_hint=coefficient_hint,
        reduction="sum",
        normalizer="none",
    )


def delta_attention_aux_loss_spec(
    *,
    coefficient_hint: float = 1.0,
) -> AuxLossSpec:
    """Return the explicit AuxLossSpec for DELTA attention regularization."""

    return AuxLossSpec(
        name="delta_attention",
        registry_key="equimo.delta_attention_loss",
        coefficient_hint=coefficient_hint,
        reduction="sum",
        normalizer="none",
    )


def task_vector_norm_loss(
    task_vector: PyTree,
    *,
    reduction: Literal["mean", "sum"] = "mean",
    coefficient: float | None = None,
) -> jax.Array:
    """Return an optionally scaled norm penalty for a task-vector delta tree."""

    delta = getattr(task_vector, "delta", task_vector)
    pairs = tuple(_norm_leaf(leaf) for _, leaf in _inexact_array_entries(delta))
    if not pairs:
        return jnp.asarray(0.0)
    sums, counts = zip(*pairs, strict=True)
    penalty = _reduce_terms(sums, counts, reduction)
    return penalty if coefficient is None else penalty * coefficient


def delta_attention_loss(
    student_features: Mapping | tuple | list | jax.Array,
    teacher_features: Mapping | tuple | list | jax.Array,
    attention_weights: Mapping | tuple | list | jax.Array,
    *,
    config: DELTAAttentionConfig | None = None,
    coefficient: float | None = None,
    reduction: Literal["sum", "mean"] | None = None,
) -> jax.Array:
    """Return DELTA attention-weighted feature-map transfer loss.

    The default layout follows the DELTA reference implementation: feature maps
    are shaped ``(batch, channels, *spatial)`` and attention weights are
    per-channel vectors.
    """

    resolved_config = DELTAAttentionConfig() if config is None else config
    selected_student = select_feature_taps(student_features, resolved_config.layers)
    selected_teacher = select_feature_taps(teacher_features, resolved_config.layers)
    selected_attention = _select_attention_taps(
        attention_weights,
        resolved_config.layers,
    )
    if not (len(selected_student) == len(selected_teacher) == len(selected_attention)):
        raise ValueError(
            "student, teacher, and attention selectors resolved different counts: "
            f"{len(selected_student)}, {len(selected_teacher)}, "
            f"{len(selected_attention)}."
        )

    resolved_reduction = resolved_config.reduction if reduction is None else reduction
    losses = tuple(
        _delta_attention_leaf(
            student,
            teacher,
            attention,
            config=resolved_config,
            reduction=resolved_reduction,
        )
        for student, teacher, attention in zip(
            selected_student,
            selected_teacher,
            selected_attention,
            strict=True,
        )
    )
    if not losses:
        return jnp.asarray(0.0)
    stacked = jnp.stack(losses)
    penalty = jnp.sum(stacked) if resolved_reduction == "sum" else jnp.mean(stacked)
    return penalty if coefficient is None else penalty * coefficient


def ewc_loss(
    model: PyTree,
    anchor: PyTree,
    fisher: PyTree,
    *,
    config: EWCConfig | None = None,
    coefficient: float | None = None,
    reduction: Literal["sum", "mean"] | None = None,
) -> jax.Array:
    """Return an optionally scaled EWC penalty using supplied Fisher diagonals."""

    resolved_config = EWCConfig() if config is None else config
    if resolved_config.fisher != "diagonal":
        raise ValueError("ewc_loss currently supports fisher='diagonal'.")
    resolved_reduction = resolved_config.reduction if reduction is None else reduction
    pairs = tuple(
        _ewc_leaf(leaf, ref, fisher_leaf)
        for leaf, ref, fisher_leaf in zip(
            _inexact_array_leaves(model),
            _inexact_array_leaves(anchor),
            _inexact_array_leaves(fisher),
            strict=True,
        )
    )
    if not pairs:
        return jnp.asarray(0.0)
    sums, counts = zip(*pairs, strict=True)
    penalty = _reduce_terms(sums, counts, resolved_reduction)
    return penalty if coefficient is None else penalty * coefficient


def feature_distillation_loss(
    student_features: jax.Array,
    teacher_features: jax.Array,
    *,
    config: FeatureDistillationConfig | None = None,
    metric: str | None = None,
    normalize_features: bool | None = None,
    stop_gradient_teacher: bool | None = None,
    reduction: Literal["mean", "sum"] | None = None,
    coefficient: float | None = None,
) -> jax.Array:
    """Return an optionally scaled feature distillation penalty."""

    resolved_config = FeatureDistillationConfig() if config is None else config
    metric = resolved_config.metric if metric is None else metric
    normalize_features = (
        resolved_config.normalize_features
        if normalize_features is None
        else normalize_features
    )
    stop_gradient_teacher = (
        resolved_config.stop_gradient_teacher
        if stop_gradient_teacher is None
        else stop_gradient_teacher
    )
    reduction = resolved_config.reduction if reduction is None else reduction

    if stop_gradient_teacher:
        teacher_features = jax.lax.stop_gradient(teacher_features)
    if normalize_features:
        student_features = _normalize(student_features)
        teacher_features = _normalize(teacher_features)

    if metric == "mse":
        values = (student_features - teacher_features) ** 2
    elif metric == "cosine":
        student = _normalize(student_features)
        teacher = _normalize(teacher_features)
        values = 1.0 - jnp.sum(student * teacher, axis=-1)
    else:
        raise ValueError(f"Unsupported feature distillation metric {metric!r}.")

    penalty = _reduce_values(values, reduction)
    return penalty if coefficient is None else penalty * coefficient


def select_feature_taps(
    taps: Mapping | tuple | list | jax.Array,
    layers: tuple[str, ...],
) -> tuple[jax.Array, ...]:
    """Select feature taps by name, index string, or percentage selector."""

    if isinstance(taps, Mapping):
        return _select_mapping_taps(taps, layers)
    if isinstance(taps, (tuple, list)):
        return _select_sequence_taps(tuple(taps), layers)
    return (jnp.asarray(taps),)


def feature_distillation_loss_from_taps(
    student_taps: Mapping | tuple | list | jax.Array,
    teacher_taps: Mapping | tuple | list | jax.Array,
    *,
    config: FeatureDistillationConfig | None = None,
    metric: str | None = None,
    normalize_features: bool | None = None,
    stop_gradient_teacher: bool | None = None,
    reduction: Literal["mean", "sum"] | None = None,
    coefficient: float | None = None,
) -> jax.Array:
    """Return feature distillation loss after selecting configured taps."""

    resolved_config = FeatureDistillationConfig() if config is None else config
    selected_student = select_feature_taps(student_taps, resolved_config.layers)
    selected_teacher = select_feature_taps(teacher_taps, resolved_config.layers)
    if len(selected_student) != len(selected_teacher):
        raise ValueError(
            "student and teacher tap selectors resolved different counts: "
            f"{len(selected_student)} and {len(selected_teacher)}."
        )

    resolved_reduction = resolved_config.reduction if reduction is None else reduction
    losses = tuple(
        feature_distillation_loss(
            student,
            teacher,
            config=resolved_config,
            metric=metric,
            normalize_features=normalize_features,
            stop_gradient_teacher=stop_gradient_teacher,
            reduction=resolved_reduction,
            coefficient=None,
        )
        for student, teacher in zip(selected_student, selected_teacher, strict=True)
    )
    if not losses:
        return jnp.asarray(0.0)
    stacked = jnp.stack(losses)
    penalty = jnp.sum(stacked) if resolved_reduction == "sum" else jnp.mean(stacked)
    return penalty if coefficient is None else penalty * coefficient


def mixout_leaf(
    leaf: jax.Array,
    anchor: jax.Array,
    *,
    key: jax.Array,
    config: MixoutConfig | None = None,
    p: float | None = None,
    inference: bool = False,
) -> jax.Array:
    """Return a Mixout-sampled leaf anchored to ``anchor``.

    The sampled value has expectation equal to ``leaf`` during training and
    returns ``leaf`` unchanged during inference.
    """

    resolved_config = MixoutConfig() if config is None else config
    p = resolved_config.p if p is None else p
    _check_mixout_p(p)
    if leaf.shape != anchor.shape:
        raise ValueError(
            f"Mixout leaf shape mismatch: got {leaf.shape} and {anchor.shape}."
        )
    if inference or p == 0.0:
        return leaf
    keep_prob = 1.0 - p
    mask = jax.random.bernoulli(key, keep_prob, leaf.shape)
    return anchor + jnp.where(mask, (leaf - anchor) / keep_prob, 0)


def mixout_tree(
    tree: PyTree,
    anchor: PyTree,
    *,
    key: jax.Array,
    config: MixoutConfig | None = None,
    p: float | None = None,
    inference: bool = False,
) -> PyTree:
    """Apply Mixout to compatible inexact array leaves in ``tree``."""

    leaves, treedef = jtu.tree_flatten(tree)
    anchor_leaves, anchor_treedef = jtu.tree_flatten(anchor)
    if treedef != anchor_treedef:
        raise ValueError("Mixout requires tree and anchor to have the same structure.")
    keys = iter(jax.random.split(key, len(leaves)))
    mixed = [
        mixout_leaf(
            leaf,
            anchor_leaf,
            key=next(keys),
            config=config,
            p=p,
            inference=inference,
        )
        if eqx.is_inexact_array(leaf) and eqx.is_inexact_array(anchor_leaf)
        else leaf
        for leaf, anchor_leaf in zip(leaves, anchor_leaves, strict=True)
    ]
    return jtu.tree_unflatten(treedef, mixed)


def _l2_leaf(leaf, ref):
    values = (leaf - ref) ** 2
    return jnp.sum(values), jnp.asarray(values.size)


def _norm_leaf(leaf):
    values = leaf**2
    return jnp.sum(values), jnp.asarray(values.size)


def _ewc_leaf(leaf, ref, fisher):
    values = fisher * (leaf - ref) ** 2
    return jnp.sum(values), jnp.asarray(values.size)


def _delta_attention_leaf(
    student: jax.Array,
    teacher: jax.Array,
    attention: jax.Array,
    *,
    config: DELTAAttentionConfig,
    reduction: Literal["sum", "mean"],
) -> jax.Array:
    student = jnp.asarray(student)
    teacher = jnp.asarray(teacher)
    if student.shape != teacher.shape:
        raise ValueError(
            "DELTA feature-map shape mismatch: "
            f"got {student.shape} and {teacher.shape}."
        )
    if config.stop_gradient_teacher:
        teacher = jax.lax.stop_gradient(teacher)

    channel_axis = _delta_channel_axis(student, config)
    diff = jnp.moveaxis(student - teacher, channel_axis, -1)
    channels = diff.shape[-1]
    attention = jnp.asarray(attention)
    if attention.shape != (channels,):
        raise ValueError(
            "DELTA attention weights must have one value per channel; "
            f"got {attention.shape}, expected ({channels},)."
        )

    spatial_axes = _delta_spatial_axes(diff)
    spatial_size = (
        math.prod(diff.shape[axis] for axis in spatial_axes) if spatial_axes else 1
    )
    squared = jnp.sum(diff**2, axis=spatial_axes) if spatial_axes else diff**2
    weighted = channels * squared * attention / spatial_size
    loss = 0.5 * jnp.sum(weighted)
    if reduction == "sum":
        return loss
    if reduction == "mean":
        return loss / jnp.maximum(jnp.asarray(weighted.size), 1)
    raise ValueError(f"Unsupported DELTA reduction {reduction!r}.")


def _delta_channel_axis(features: jax.Array, config: DELTAAttentionConfig) -> int:
    raw_axis = config.feature_alignment.get("channel_axis")
    if raw_axis is None:
        raw_axis = 1 if features.ndim >= 4 else 0
    axis = int(raw_axis)
    if axis < 0:
        axis += features.ndim
    if not 0 <= axis < features.ndim:
        raise ValueError(
            f"DELTA channel_axis {raw_axis!r} is out of range for shape {features.shape}."
        )
    return axis


def _delta_spatial_axes(features_channel_last: jax.Array) -> tuple[int, ...]:
    if features_channel_last.ndim <= 2:
        return ()
    if features_channel_last.ndim >= 4:
        return tuple(range(1, features_channel_last.ndim - 1))
    return tuple(range(0, features_channel_last.ndim - 1))


def _matched_l2_entries(
    model: PyTree,
    reference: PyTree,
    config: L2SPConfig,
) -> tuple[tuple[Path, jax.Array, jax.Array], ...]:
    model_entries = _inexact_array_entries(model)
    reference_entries = _inexact_array_entries(reference)
    if len(model_entries) != len(reference_entries):
        raise ValueError(
            "l2_sp_loss requires compatible inexact-array tree structures."
        )

    matched: list[tuple[Path, jax.Array, jax.Array]] = []
    for (path, leaf), (ref_path, ref) in zip(
        model_entries, reference_entries, strict=True
    ):
        if path != ref_path:
            raise ValueError(
                "l2_sp_loss requires matching inexact-array paths; "
                f"got {path_to_str(path)} and {path_to_str(ref_path)}."
            )
        matched.append((path, leaf, ref))
    return tuple(matched)


def _include_l2_path(path: Path, leaf: jax.Array, config: L2SPConfig) -> bool:
    if config.shared_mask not in {"all", "backbone", "transferred_trainable"}:
        raise ValueError(
            "L2SPConfig.shared_mask must be 'all', 'backbone', or "
            "'transferred_trainable'."
        )
    if config.shared_mask == "all":
        return True
    tags = canonical_tags_for_path(path, leaf)
    excluded = {
        "head",
        "lora",
        "dora",
        "adapter",
        "prompt",
        "prefix",
        "ia3",
        "scale_shift",
    }
    return tags.isdisjoint(excluded)


def _include_l2_new_path(path: Path, leaf: jax.Array, config: L2SPConfig) -> bool:
    if config.beta == 0.0:
        return False
    if config.new_mask not in {"none", "head", "newly_initialized", "all"}:
        raise ValueError(
            "L2SPConfig.new_mask must be 'none', 'head', 'newly_initialized', or 'all'."
        )
    if config.new_mask == "none":
        return False
    if config.new_mask == "all":
        return True
    tags = canonical_tags_for_path(path, leaf)
    path_parts = {str(part).strip("[]'\"") for part in path}
    return "head" in tags or "head" in path_parts or "head" in path_to_str(path)


def _inexact_array_entries(tree: PyTree) -> tuple[tuple[Path, jax.Array], ...]:
    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(tree)
        if eqx.is_inexact_array(leaf)
    )


def _inexact_array_leaves(tree: PyTree) -> tuple[jax.Array, ...]:
    return tuple(leaf for leaf in jtu.tree_leaves(tree) if eqx.is_inexact_array(leaf))


def _reduce_terms(
    sums: tuple[jax.Array, ...],
    counts: tuple[jax.Array, ...],
    reduction: Literal["mean", "sum"],
) -> jax.Array:
    total = jnp.sum(jnp.stack(sums))
    if reduction == "sum":
        return total
    if reduction == "mean":
        count = jnp.sum(jnp.stack(counts))
        return total / jnp.maximum(count, 1)
    raise ValueError(f"Unsupported reduction {reduction!r}.")


def _reduce_l2sp_terms(
    sums: tuple[jax.Array, ...],
    counts: tuple[jax.Array, ...],
    reduction: Literal["sum", "mean_per_parameter"],
) -> jax.Array:
    total = jnp.sum(jnp.stack(sums))
    if reduction == "sum":
        return total
    if reduction == "mean_per_parameter":
        count = jnp.sum(jnp.stack(counts))
        return total / jnp.maximum(count, 1)
    raise ValueError(f"Unsupported L2-SP reduction {reduction!r}.")


def _reduce_l2sp_terms_from_pairs(
    pairs: tuple[tuple[jax.Array, jax.Array], ...],
    reduction: Literal["sum", "mean_per_parameter"],
) -> jax.Array:
    if not pairs:
        return jnp.asarray(0.0)
    sums, counts = zip(*pairs, strict=True)
    return _reduce_l2sp_terms(sums, counts, reduction)


def _reduce_values(
    values: jax.Array,
    reduction: Literal["mean", "sum"],
) -> jax.Array:
    if reduction == "mean":
        return jnp.mean(values)
    if reduction == "sum":
        return jnp.sum(values)
    raise ValueError(f"Unsupported reduction {reduction!r}.")


def _normalize(x: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, 1e-12)


def _select_mapping_taps(
    taps: Mapping,
    layers: tuple[str, ...],
) -> tuple[jax.Array, ...]:
    if not taps:
        raise ValueError("Feature tap mapping is empty.")

    items = tuple(taps.items())
    values = tuple(value for _, value in items)
    selected: list[jax.Array] = []
    for layer in layers:
        if layer in taps:
            selected.append(jnp.asarray(taps[layer]))
            continue
        if layer.isdigit() and int(layer) in taps:
            selected.append(jnp.asarray(taps[int(layer)]))
            continue
        if _is_percentage_selector(layer):
            selected.append(jnp.asarray(values[_percentage_index(layer, len(values))]))
            continue
        raise ValueError(f"Feature tap selector {layer!r} was not found.")
    return tuple(selected)


def _select_sequence_taps(
    taps: tuple,
    layers: tuple[str, ...],
) -> tuple[jax.Array, ...]:
    if not taps:
        raise ValueError("Feature tap sequence is empty.")

    selected: list[jax.Array] = []
    for layer in layers:
        if layer.isdigit():
            index = int(layer)
            if not 0 <= index < len(taps):
                raise ValueError(
                    f"Feature tap index {index} is out of range for {len(taps)} taps."
                )
            selected.append(jnp.asarray(taps[index]))
            continue
        if _is_percentage_selector(layer):
            selected.append(jnp.asarray(taps[_percentage_index(layer, len(taps))]))
            continue
        raise ValueError(
            f"Feature tap selector {layer!r} requires named taps, "
            "but a sequence was provided."
        )
    return tuple(selected)


def _select_attention_taps(
    attention: Mapping | tuple | list | jax.Array,
    layers: tuple[str, ...],
) -> tuple[jax.Array, ...]:
    if isinstance(attention, Mapping):
        return select_feature_taps(attention, layers)
    if isinstance(attention, (tuple, list)):
        if len(attention) == len(layers):
            return tuple(jnp.asarray(item) for item in attention)
        return select_feature_taps(tuple(attention), layers)
    if len(layers) != 1:
        raise ValueError("A single DELTA attention vector requires exactly one layer.")
    return (jnp.asarray(attention),)


def _is_percentage_selector(selector: str) -> bool:
    if not selector.endswith("%"):
        return False
    try:
        float(selector[:-1])
    except ValueError:
        return False
    return True


def _percentage_index(selector: str, length: int) -> int:
    value = float(selector[:-1])
    if not 0.0 < value <= 100.0:
        raise ValueError("Percentage feature tap selectors must be in (0%, 100%].")
    return min(length - 1, max(0, math.ceil(value / 100.0 * length) - 1))


def _check_mixout_p(p: float) -> None:
    if not 0.0 <= p < 1.0:
        raise ValueError("Mixout probability p must satisfy 0 <= p < 1.")


__all__ = (
    "DELTAAttentionConfig",
    "EWCConfig",
    "FeatureDistillationConfig",
    "L2SPConfig",
    "MixoutConfig",
    "adalora_orthogonality_aux_loss_spec",
    "adalora_orthogonality_loss",
    "adapter_norm_loss",
    "delta_attention_aux_loss_spec",
    "delta_attention_loss",
    "ewc_loss",
    "feature_distillation_loss",
    "feature_distillation_loss_from_taps",
    "l2_sp_loss",
    "mixout_leaf",
    "mixout_tree",
    "select_feature_taps",
    "task_vector_norm_loss",
)
