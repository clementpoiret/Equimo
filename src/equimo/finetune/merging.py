"""Model merging and task-vector utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ._typing import PyTree
from .config import FineTuneBundle, ModelLineage
from .paths import key_path_to_path, path_to_str


@dataclass(frozen=True)
class WiSEFTConfig:
    """Configuration for WiSE-FT interpolation."""

    alpha: float = 0.5
    head_policy: str = "exclude"
    mask: str = "shared_backbone"
    include_head: bool = False
    strict_shapes: bool = True
    require_same_architecture_hash: bool = True


@dataclass(frozen=True)
class UniformSoupConfig:
    """Configuration for uniform model soups."""

    weights: tuple[float, ...] | str = "equal"
    require_same_base_hash: bool = True
    strict_shapes: bool = True
    mask: str = "floating_compatible_leaves"


@dataclass(frozen=True)
class GreedySoupConfig:
    """Configuration metadata for greedy model soups."""

    sort_by: str = "validation_score_desc"
    start: str = "best_model"
    add_if_improves: bool = True
    min_delta: float = 0.0
    require_same_base_hash: bool = True


@dataclass(frozen=True)
class TaskVectorConfig:
    """Configuration for task-vector extraction/application."""

    scale: float = 1.0
    mask: str = "floating_backbone"
    include_head: bool = False
    require_same_base_hash: bool = True


@dataclass(frozen=True)
class TaskVector:
    """Difference PyTree from a base model to a fine-tuned model."""

    delta: PyTree
    include_head: bool = False
    base_architecture_hash: str = ""
    base_checkpoint_hash: str = ""
    logical_id_table_hash: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MergePlan:
    """Prepared modern merge operation."""

    method: str
    task_vectors: tuple[TaskVector, ...]
    method_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MergeMethod(Protocol):
    """Common interface for data-aware or subspace-aware model mergers."""

    def prepare(
        self,
        models: Sequence[PyTree | TaskVector],
        artifacts: Mapping[str, Any] | None = None,
    ) -> MergePlan:
        """Prepare a merge plan from models or precomputed task vectors."""

    def merge(self, plan: MergePlan) -> FineTuneBundle:
        """Merge a prepared plan into a portable fine-tuning bundle."""


@dataclass(frozen=True)
class TIESConfig:
    """TIES merge metadata."""

    density: float = 0.20
    density_scope: str = "global"
    trim_by: str = "magnitude"
    sign_election: str = "sum"
    zero_sign: str = "zero"
    merge: str = "disjoint_mean"
    final_scale: float = 1.0
    require_same_base_hash: bool = True


@dataclass(frozen=True)
class DARETransform:
    """DARE merge metadata."""

    drop_rate: float = 0.90
    rescale: bool = True
    seed: int | None = None
    scope: str = "per_tensor"


@dataclass(frozen=True)
class BreadcrumbsConfig:
    """Model Breadcrumbs merge metadata."""

    bottom_fraction: float = 0.05
    top_fraction: float = 0.01
    rescale: bool = False
    require_same_base_hash: bool = True


@dataclass(frozen=True)
class FisherMergeConfig:
    """Fisher merge metadata."""

    fisher: str = "diagonal"
    normalize_fisher: bool = True
    eps: float = 1e-8
    require_statistics: bool = True


@dataclass(frozen=True)
class RegMeanConfig:
    """RegMean merge metadata."""

    ridge: float = 1e-5
    covariance_normalization: str = "none"
    solver: str = "solve"
    non_matrix_policy: str = "mean"
    require_input_covariances: bool = True
    require_same_architecture_hash: bool = True


@dataclass(frozen=True)
class KnOTSConfig:
    """KnOTS shared-left-basis merge metadata."""

    rank: int | None = None
    orientation: str = "out_in"
    basis_policy: str = "shared_left_svd"
    non_matrix_policy: str = "mean"
    require_same_base_hash: bool = True


def interpolate_models(
    base_model: PyTree,
    tuned_model: PyTree,
    *,
    alpha: float = 0.5,
    include_head: bool = False,
    config: WiSEFTConfig | None = None,
) -> PyTree:
    """Interpolate compatible model leaves."""

    strict_shapes = True
    head_policy = "interpolate_compatible" if include_head else "exclude"
    if config is not None:
        alpha = config.alpha
        include_head = config.include_head
        strict_shapes = config.strict_shapes
        head_policy = config.head_policy
        _validate_wise_head_policy(head_policy)
        if include_head and head_policy == "exclude":
            head_policy = "interpolate_compatible"
        include_head = include_head or head_policy in {
            "interpolate_compatible",
            "use_finetuned",
        }
        if config.require_same_architecture_hash:
            _check_same_architecture(
                (base_model, tuned_model),
                include_head=head_policy == "interpolate_compatible" and include_head,
            )

    return jtu.tree_map_with_path(
        lambda path, base, tuned: _interpolate_leaf(
            key_path_to_path(path),
            base,
            tuned,
            alpha=alpha,
            include_head=include_head,
            head_policy=head_policy,
            strict_shapes=strict_shapes,
        ),
        base_model,
        tuned_model,
    )


def uniform_soup(
    models: Sequence[PyTree],
    *,
    weights: Sequence[float] | str = "equal",
    include_head: bool = True,
    config: UniformSoupConfig | None = None,
) -> PyTree:
    """Return the weighted arithmetic mean of compatible model leaves."""

    if not models:
        raise ValueError("uniform_soup requires at least one model.")
    strict_shapes = True
    if config is not None:
        weights = config.weights
        strict_shapes = config.strict_shapes
        _check_same_base_metadata(
            models,
            require_same_base_hash=config.require_same_base_hash,
        )
        if config.strict_shapes:
            _check_same_architecture(models, include_head=include_head)
    resolved_weights = _resolve_weights(len(models), weights)
    first, *rest = models
    return jtu.tree_map_with_path(
        lambda path, leaf, *other_leaves: _soup_leaf(
            key_path_to_path(path),
            (leaf, *other_leaves),
            resolved_weights,
            include_head=include_head,
            strict_shapes=strict_shapes,
        ),
        first,
        *rest,
    )


def greedy_soup(
    models: Sequence[PyTree],
    score_fn: Callable[[PyTree], float],
    *,
    higher_is_better: bool = True,
    config: GreedySoupConfig | None = None,
) -> tuple[PyTree, tuple[int, ...]]:
    """Greedily add models when the user-provided score improves."""

    if not models:
        raise ValueError("greedy_soup requires at least one model.")
    min_delta = 0.0 if config is None else config.min_delta
    add_if_improves = True if config is None else config.add_if_improves
    if config is not None:
        _check_same_base_metadata(
            models,
            require_same_base_hash=config.require_same_base_hash,
        )
        if config.sort_by != "validation_score_desc":
            raise ValueError(
                "GreedySoupConfig.sort_by currently supports "
                "'validation_score_desc'."
            )
        if config.start != "best_model":
            raise ValueError(
                "GreedySoupConfig.start currently supports 'best_model'."
            )
        return _greedy_soup_from_best(
            models,
            score_fn,
            higher_is_better=higher_is_better,
            min_delta=min_delta,
            add_if_improves=add_if_improves,
        )

    selected = [0]
    soup = models[0]
    best_score = score_fn(soup)

    for index, candidate in enumerate(models[1:], start=1):
        proposal = uniform_soup([soup, candidate])
        score = score_fn(proposal)
        improves = (
            score >= best_score + min_delta
            if higher_is_better
            else score <= best_score - min_delta
        )
        if add_if_improves and improves:
            soup = proposal
            best_score = score
            selected.append(index)

    return soup, tuple(selected)


def task_vector(
    base_model: PyTree,
    tuned_model: PyTree,
    *,
    include_head: bool = False,
    config: TaskVectorConfig | None = None,
) -> TaskVector:
    """Extract a task vector from compatible model leaves."""

    if config is not None:
        include_head = config.include_head
        if config.require_same_base_hash:
            _check_same_architecture(
                (base_model, tuned_model),
                include_head=include_head,
            )

    delta = jtu.tree_map_with_path(
        lambda path, base, tuned: _delta_leaf(
            key_path_to_path(path),
            base,
            tuned,
            include_head=include_head,
        ),
        base_model,
        tuned_model,
    )
    return TaskVector(
        delta=delta,
        include_head=include_head,
        base_architecture_hash=_architecture_hash(base_model, include_head=include_head),
        base_checkpoint_hash=_checkpoint_hash(base_model, include_head=include_head),
        logical_id_table_hash=_logical_id_table_hash(
            base_model,
            include_head=include_head,
        ),
    )


def apply_task_vector(
    base_model: PyTree,
    vector: TaskVector,
    *,
    scale: float = 1.0,
    config: TaskVectorConfig | None = None,
) -> PyTree:
    """Apply a task vector to a compatible base model."""

    if config is not None:
        scale = config.scale
    _check_task_vector_base(base_model, vector)
    return jtu.tree_map(
        lambda base, delta: base + delta * scale
        if eqx.is_inexact_array(base) and eqx.is_inexact_array(delta)
        else base,
        base_model,
        vector.delta,
    )


def ties_merge(
    vectors: Sequence[TaskVector],
    *,
    density: float = 0.20,
    config: TIESConfig | None = None,
) -> TaskVector:
    """Merge task vectors with a simple TIES-style sign consensus."""

    if not vectors:
        raise ValueError("ties_merge requires at least one task vector.")
    final_scale = 1.0
    density_scope = "global"
    if config is not None:
        density = config.density
        final_scale = config.final_scale
        density_scope = config.density_scope
        if config.density_scope not in {"global", "per_tensor"}:
            raise ValueError(
                "TIESConfig.density_scope must be 'global' or 'per_tensor'."
            )
        if config.trim_by != "magnitude":
            raise ValueError("TIESConfig.trim_by currently supports 'magnitude'.")
        if config.sign_election != "sum":
            raise ValueError("TIESConfig.sign_election currently supports 'sum'.")
        if config.merge != "disjoint_mean":
            raise ValueError("TIESConfig.merge currently supports 'disjoint_mean'.")
        _check_vector_compatibility(
            vectors,
            require_same_base_hash=config.require_same_base_hash,
        )
    else:
        _check_vector_compatibility(vectors, require_same_base_hash=True)
    if density_scope == "global":
        deltas = [_trim_delta_global(vector.delta, density) for vector in vectors]
        density = 1.0
    else:
        deltas = [vector.delta for vector in vectors]
    merged = _merge_deltas(
        deltas,
        lambda leaves: _ties_leaf(
            leaves,
            density,
            zero_sign="zero" if config is None else config.zero_sign,
            final_scale=final_scale,
        ),
    )
    first = vectors[0]
    return TaskVector(
        merged,
        include_head=all(vector.include_head for vector in vectors),
        base_architecture_hash=first.base_architecture_hash,
        base_checkpoint_hash=first.base_checkpoint_hash,
        logical_id_table_hash=first.logical_id_table_hash,
    )


def dare_task_vector(
    vector: TaskVector,
    *,
    drop_rate: float = 0.90,
    key: jax.Array | None = None,
    rescale: bool = True,
    config: DARETransform | None = None,
) -> TaskVector:
    """Apply a static DARE mask to a task vector."""

    scope = "per_tensor"
    if config is not None:
        drop_rate = config.drop_rate
        rescale = config.rescale
        scope = config.scope
        if scope not in {"global", "per_tensor"}:
            raise ValueError("DARETransform.scope must be 'global' or 'per_tensor'.")
        if config.seed is not None:
            if key is not None:
                raise ValueError("Pass either key or DARETransform.seed, not both.")
            key = jax.random.PRNGKey(config.seed)
    if key is None:
        raise ValueError("dare_task_vector requires key or DARETransform.seed.")
    if not 0.0 <= drop_rate <= 1.0:
        raise ValueError("DARE drop_rate must satisfy 0 <= drop_rate <= 1.")
    if scope == "global":
        delta = _dare_delta_global(
            vector.delta,
            drop_rate=drop_rate,
            key=key,
            rescale=rescale,
        )
        return TaskVector(
            delta,
            include_head=vector.include_head,
            base_architecture_hash=vector.base_architecture_hash,
            base_checkpoint_hash=vector.base_checkpoint_hash,
            logical_id_table_hash=vector.logical_id_table_hash,
        )
    leaves, treedef = jtu.tree_flatten(vector.delta)
    keys = jax.random.split(key, len(leaves))
    masked = [
        _dare_leaf(leaf, drop_rate=drop_rate, key=leaf_key, rescale=rescale)
        for leaf, leaf_key in zip(leaves, keys, strict=True)
    ]
    return TaskVector(
        jtu.tree_unflatten(treedef, masked),
        include_head=vector.include_head,
        base_architecture_hash=vector.base_architecture_hash,
        base_checkpoint_hash=vector.base_checkpoint_hash,
        logical_id_table_hash=vector.logical_id_table_hash,
    )


def breadcrumbs_task_vector(
    vector: TaskVector,
    *,
    bottom_fraction: float = 0.05,
    top_fraction: float = 0.01,
    config: BreadcrumbsConfig | None = None,
) -> TaskVector:
    """Remove the smallest and largest task-vector deltas by magnitude."""

    if config is not None:
        bottom_fraction = config.bottom_fraction
        top_fraction = config.top_fraction
        if config.rescale:
            raise ValueError("BreadcrumbsConfig.rescale currently supports False.")
    _validate_breadcrumbs_fractions(bottom_fraction, top_fraction)
    return TaskVector(
        jtu.tree_map(
            lambda leaf: _breadcrumbs_leaf(
                leaf,
                bottom_fraction=bottom_fraction,
                top_fraction=top_fraction,
            )
            if eqx.is_inexact_array(leaf)
            else leaf,
            vector.delta,
        ),
        include_head=vector.include_head,
        base_architecture_hash=vector.base_architecture_hash,
        base_checkpoint_hash=vector.base_checkpoint_hash,
        logical_id_table_hash=vector.logical_id_table_hash,
    )


def fisher_merge(
    models: Sequence[PyTree],
    fishers: Sequence[PyTree] | None = None,
    *,
    eps: float = 1e-8,
    config: FisherMergeConfig | None = None,
) -> PyTree:
    """Fisher-weighted merge requiring external Fisher statistics."""

    normalize_fisher = True
    if config is not None:
        eps = config.eps
        normalize_fisher = config.normalize_fisher
        if config.fisher != "diagonal":
            raise ValueError("fisher_merge currently supports fisher='diagonal'.")
    if fishers is None:
        raise ValueError("fisher_merge requires external Fisher statistics.")
    if len(models) != len(fishers):
        raise ValueError("models and fishers must have the same length.")
    resolved_fishers = (
        tuple(_normalize_fisher_tree(fisher) for fisher in fishers)
        if normalize_fisher
        else tuple(fishers)
    )
    first_model, *other_models = models
    first_fisher, *other_fishers = resolved_fishers
    return jtu.tree_map(
        lambda model_leaf, *rest: _fisher_leaf(
            (model_leaf, *rest[: len(other_models)]),
            (rest[len(other_models)], *rest[len(other_models) + 1 :]),
            eps=eps,
        ),
        first_model,
        *other_models,
        first_fisher,
        *other_fishers,
    )


def regmean_merge(
    models: Sequence[PyTree],
    covariances: Sequence[PyTree] | None = None,
    *,
    ridge: float = 1e-5,
    config: RegMeanConfig | None = None,
) -> PyTree:
    """RegMean-style merge requiring external covariance statistics."""

    non_matrix_policy = "mean"
    solver = "solve"
    if config is not None:
        ridge = config.ridge
        non_matrix_policy = config.non_matrix_policy
        solver = config.solver
        if config.covariance_normalization != "none":
            raise ValueError(
                "RegMeanConfig.covariance_normalization currently supports 'none'."
            )
        if config.solver not in {"cholesky", "solve", "svd"}:
            raise ValueError(
                "RegMeanConfig.solver must be 'cholesky', 'solve', or 'svd'."
            )
        if config.non_matrix_policy == "base":
            raise ValueError(
                "RegMeanConfig.non_matrix_policy='base' requires a base model."
            )
        if config.non_matrix_policy not in {"mean", "error"}:
            raise ValueError(
                "RegMeanConfig.non_matrix_policy must be 'mean', 'base', or 'error'."
            )
        if config.require_same_architecture_hash:
            _check_same_architecture(models, include_head=True)
    if covariances is None:
        raise ValueError("regmean_merge requires external covariance statistics.")
    if len(models) != len(covariances):
        raise ValueError("models and covariances must have the same length.")
    first_model, *other_models = models
    first_covariance, *other_covariances = covariances
    return jtu.tree_map(
        lambda model_leaf, *rest: _regmean_leaf(
            (model_leaf, *rest[: len(other_models)]),
            (rest[len(other_models)], *rest[len(other_models) + 1 :]),
            ridge=ridge,
            non_matrix_policy=non_matrix_policy,
            solver=solver,
        ),
        first_model,
        *other_models,
        first_covariance,
        *other_covariances,
    )


def knots_task_vector(
    vectors: Sequence[TaskVector],
    *,
    config: KnOTSConfig | None = None,
) -> TaskVector:
    """Merge task vectors with a KnOTS-style shared left singular basis."""

    if not vectors:
        raise ValueError("knots_task_vector requires at least one task vector.")
    config = KnOTSConfig() if config is None else config
    if config.rank is not None and config.rank <= 0:
        raise ValueError("KnOTS rank must be positive when provided.")
    if config.orientation != "out_in":
        raise ValueError("KnOTSConfig.orientation currently supports 'out_in'.")
    if config.basis_policy != "shared_left_svd":
        raise ValueError("KnOTSConfig.basis_policy currently supports 'shared_left_svd'.")
    _check_vector_compatibility(
        vectors,
        require_same_base_hash=config.require_same_base_hash,
    )
    merged = _merge_deltas(
        [vector.delta for vector in vectors],
        lambda leaves: _knots_leaf(leaves, config=config),
    )
    return _merged_task_vector(vectors, merged, metadata=_knots_metadata(config))


def task_vector_bundle(
    vector: TaskVector,
    *,
    method: str,
    metadata: Mapping[str, Any] | None = None,
    adapter_config: Mapping[str, Any] | None = None,
) -> FineTuneBundle:
    """Package a merged task vector as a portable fine-tuning bundle."""

    merged_metadata = dict(vector.metadata)
    if metadata:
        merged_metadata.update(metadata)
    method_config = dict(adapter_config or {})
    method_config.setdefault("method", method)
    method_config.setdefault("include_head", vector.include_head)
    method_config.setdefault("metadata", merged_metadata)
    return FineTuneBundle(
        method=method,
        schema_version=1,
        architecture_hash=vector.base_architecture_hash,
        adapter_config=method_config,
        delta_tree=vector.delta,
        lineage=ModelLineage(
            architecture_hash=vector.base_architecture_hash or None,
            base_checkpoint_hash=vector.base_checkpoint_hash or None,
            base_value_hash=vector.base_checkpoint_hash or None,
            logical_id_table_hash=vector.logical_id_table_hash or None,
            notes={"merge": merged_metadata},
        ),
        metadata=merged_metadata,
    )


@dataclass(frozen=True)
class KnOTSMerging:
    """KnOTS method wrapper for the common merge interface."""

    config: KnOTSConfig = field(default_factory=KnOTSConfig)

    def prepare(
        self,
        models: Sequence[PyTree | TaskVector],
        artifacts: Mapping[str, Any] | None = None,
    ) -> MergePlan:
        vectors = _task_vectors_from_inputs(models, artifacts)
        method_config = _knots_config_dict(self.config)
        return MergePlan(
            method="knots",
            task_vectors=vectors,
            method_config=method_config,
            metadata=_knots_metadata(self.config),
        )

    def merge(self, plan: MergePlan) -> FineTuneBundle:
        config = _knots_config_from_mapping(plan.method_config)
        vector = knots_task_vector(plan.task_vectors, config=config)
        return task_vector_bundle(
            vector,
            method=plan.method,
            metadata=plan.metadata,
            adapter_config={"method_config": plan.method_config},
        )


def _task_vectors_from_inputs(
    models: Sequence[PyTree | TaskVector],
    artifacts: Mapping[str, Any] | None,
) -> tuple[TaskVector, ...]:
    artifacts = {} if artifacts is None else artifacts
    artifact_vectors = artifacts.get("task_vectors")
    if artifact_vectors is not None:
        return _task_vector_tuple(artifact_vectors)
    if all(isinstance(model, TaskVector) for model in models):
        return _task_vector_tuple(models)
    base_model = artifacts.get("base_model")
    if base_model is not None:
        include_head = bool(artifacts.get("include_head", False))
        return tuple(
            task_vector(base_model, model, include_head=include_head)
            for model in models
        )
    raise ValueError(
        "Modern merge preparation requires TaskVector inputs, "
        "artifacts['task_vectors'], or artifacts['base_model']."
    )


def _task_vector_tuple(vectors: Sequence[Any]) -> tuple[TaskVector, ...]:
    resolved = tuple(vectors)
    if not resolved:
        raise ValueError("Modern merge preparation requires at least one task vector.")
    if not all(isinstance(vector, TaskVector) for vector in resolved):
        raise TypeError("task_vectors must contain only TaskVector instances.")
    return resolved


def _merged_task_vector(
    vectors: Sequence[TaskVector],
    merged: PyTree,
    *,
    metadata: Mapping[str, Any],
) -> TaskVector:
    first = vectors[0]
    return TaskVector(
        merged,
        include_head=all(vector.include_head for vector in vectors),
        base_architecture_hash=first.base_architecture_hash,
        base_checkpoint_hash=first.base_checkpoint_hash,
        logical_id_table_hash=first.logical_id_table_hash,
        metadata=dict(metadata),
    )


def _weighted_leaf(leaves, weights: Sequence[float]):
    first = leaves[0]
    if not eqx.is_inexact_array(first):
        return first
    _check_leaf_shapes(leaves)
    total = jnp.zeros_like(first)
    for leaf, weight in zip(leaves, weights, strict=True):
        total = total + leaf * weight
    return total


def _knots_leaf(leaves, *, config: KnOTSConfig):
    first = leaves[0]
    if not eqx.is_inexact_array(first):
        return first
    _check_leaf_shapes(leaves)
    if first.ndim != 2:
        return _non_matrix_leaf(leaves, config.non_matrix_policy)
    concatenated = jnp.concatenate(tuple(jnp.asarray(leaf) for leaf in leaves), axis=1)
    left_basis, _, _ = jnp.linalg.svd(concatenated, full_matrices=False)
    rank = _resolve_svd_rank(config.rank, left_basis.shape[1])
    if rank == 0:
        return jnp.zeros_like(first)
    basis = left_basis[:, :rank]
    coordinate_sum = jnp.zeros((rank, first.shape[1]), dtype=first.dtype)
    for leaf in leaves:
        coordinate_sum = coordinate_sum + basis.T @ leaf
    return basis @ (coordinate_sum / len(leaves))


def _non_matrix_leaf(leaves, policy: str):
    if policy == "mean":
        return _weighted_leaf(leaves, (1.0 / len(leaves),) * len(leaves))
    if policy == "base":
        first = leaves[0]
        return jnp.zeros_like(first) if eqx.is_inexact_array(first) else first
    if policy == "error":
        raise ValueError("Modern merge encountered a non-matrix leaf.")
    raise ValueError("non_matrix_policy must be 'mean', 'base', or 'error'.")


def _check_leaf_shapes(leaves) -> None:
    first = leaves[0]
    for leaf in leaves:
        if not eqx.is_inexact_array(leaf) or leaf.shape != first.shape:
            raise ValueError("Modern merge encountered incompatible leaves.")


def _resolve_svd_rank(rank: int | None, max_rank: int) -> int:
    if max_rank <= 0:
        return 0
    if rank is None:
        return int(max_rank)
    return min(int(rank), int(max_rank))


def _knots_metadata(config: KnOTSConfig) -> dict[str, Any]:
    return {
        "method": "knots",
        "profile_fidelity": "experimental",
        "rank": config.rank,
        "basis_policy": config.basis_policy,
        "orientation": config.orientation,
        "non_matrix_policy": config.non_matrix_policy,
    }


def _knots_config_dict(config: KnOTSConfig) -> dict[str, Any]:
    return {
        "rank": config.rank,
        "orientation": config.orientation,
        "basis_policy": config.basis_policy,
        "non_matrix_policy": config.non_matrix_policy,
        "require_same_base_hash": config.require_same_base_hash,
    }


def _knots_config_from_mapping(config: Mapping[str, Any]) -> KnOTSConfig:
    return KnOTSConfig(
        rank=config.get("rank"),
        orientation=str(config.get("orientation", "out_in")),
        basis_policy=str(config.get("basis_policy", "shared_left_svd")),
        non_matrix_policy=str(config.get("non_matrix_policy", "mean")),
        require_same_base_hash=bool(config.get("require_same_base_hash", True)),
    )


def _greedy_soup_from_best(
    models: Sequence[PyTree],
    score_fn: Callable[[PyTree], float],
    *,
    higher_is_better: bool,
    min_delta: float,
    add_if_improves: bool,
) -> tuple[PyTree, tuple[int, ...]]:
    scores = tuple(float(score_fn(model)) for model in models)
    order = tuple(
        sorted(
            range(len(models)),
            key=lambda index: scores[index],
            reverse=higher_is_better,
        )
    )
    selected = [order[0]]
    soup = models[order[0]]
    best_score = scores[order[0]]

    for index in order[1:]:
        candidate = models[index]
        proposal = uniform_soup([soup, candidate])
        score = float(score_fn(proposal))
        improves = (
            score >= best_score + min_delta
            if higher_is_better
            else score <= best_score - min_delta
        )
        if add_if_improves and improves:
            soup = proposal
            best_score = score
            selected.append(index)

    return soup, tuple(selected)


def _validate_wise_head_policy(head_policy: str) -> None:
    if head_policy not in {
        "interpolate_compatible",
        "use_zero_shot",
        "use_finetuned",
        "exclude",
    }:
        raise ValueError(
            "WiSEFTConfig.head_policy must be 'interpolate_compatible', "
            "'use_zero_shot', 'use_finetuned', or 'exclude'."
        )


def _interpolate_leaf(
    path,
    base,
    tuned,
    *,
    alpha: float,
    include_head: bool,
    head_policy: str,
    strict_shapes: bool,
):
    if _is_head_path(path):
        if head_policy in {"exclude", "use_zero_shot"}:
            return base
        if head_policy == "use_finetuned":
            return tuned if eqx.is_inexact_array(tuned) else base
    if not _mergeable(
        path,
        base,
        tuned,
        include_head=include_head,
        strict_shapes=strict_shapes,
    ):
        return base
    return (1.0 - alpha) * base + alpha * tuned


def _soup_leaf(path, leaves, weights, *, include_head: bool, strict_shapes: bool = True):
    first = leaves[0]
    if not eqx.is_inexact_array(first) or (not include_head and _is_head_path(path)):
        return first
    total = jnp.zeros_like(first)
    for leaf, weight in zip(leaves, weights, strict=True):
        if not eqx.is_inexact_array(leaf) or leaf.shape != first.shape:
            if strict_shapes:
                raise ValueError("uniform_soup encountered incompatible leaves.")
            return first
        total = total + leaf * weight
    return total


def _delta_leaf(path, base, tuned, *, include_head: bool):
    if not _mergeable(path, base, tuned, include_head=include_head):
        return jnp.zeros_like(base) if eqx.is_inexact_array(base) else base
    return tuned - base


def _mergeable(path, base, tuned, *, include_head: bool, strict_shapes: bool = True) -> bool:
    if not eqx.is_inexact_array(base) or not eqx.is_inexact_array(tuned):
        return False
    if not include_head and _is_head_path(path):
        return False
    if base.shape != tuned.shape:
        if not strict_shapes:
            return False
        raise ValueError("Cannot merge leaves with different shapes.")
    return True


def _resolve_weights(count: int, weights: Sequence[float] | str) -> tuple[float, ...]:
    if weights == "equal":
        return (1.0 / count,) * count
    if len(weights) != count:
        raise ValueError("weights length must match model count.")
    total = float(sum(weights))
    if total == 0.0:
        raise ValueError("weights must not sum to zero.")
    return tuple(float(weight) / total for weight in weights)


def _merge_deltas(deltas: Sequence[PyTree], leaf_fn) -> PyTree:
    first, *rest = deltas
    return jtu.tree_map(lambda leaf, *others: leaf_fn((leaf, *others)), first, *rest)


def _trim_delta_global(delta: PyTree, density: float) -> PyTree:
    if not 0.0 < density <= 1.0:
        raise ValueError("TIES density must satisfy 0 < density <= 1.")
    if density == 1.0:
        return delta
    leaves, treedef = jtu.tree_flatten(delta)
    flat_abs_parts = [
        jnp.abs(leaf).reshape(-1)
        for leaf in leaves
        if eqx.is_inexact_array(leaf) and leaf.size > 0
    ]
    if not flat_abs_parts:
        return delta
    threshold = jnp.quantile(jnp.concatenate(flat_abs_parts), 1.0 - density)
    trimmed = [
        jnp.where(jnp.abs(leaf) >= threshold, leaf, 0)
        if eqx.is_inexact_array(leaf)
        else leaf
        for leaf in leaves
    ]
    return jtu.tree_unflatten(treedef, trimmed)


def _ties_leaf(
    leaves,
    density: float,
    *,
    zero_sign: str,
    final_scale: float,
):
    first = leaves[0]
    if not eqx.is_inexact_array(first):
        return first
    stacked = jnp.stack(leaves)
    if not 0.0 < density <= 1.0:
        raise ValueError("TIES density must satisfy 0 < density <= 1.")
    if density < 1.0:
        thresholds = jnp.quantile(
            jnp.abs(stacked).reshape((stacked.shape[0], -1)),
            1.0 - density,
            axis=1,
        )
        trim_mask = jnp.abs(stacked) >= thresholds.reshape((-1,) + (1,) * first.ndim)
        stacked = jnp.where(trim_mask, stacked, 0)
    sign_sum = jnp.sum(stacked, axis=0)
    consensus = jnp.sign(sign_sum)
    if zero_sign == "positive":
        consensus = jnp.where(consensus == 0, 1, consensus)
    elif zero_sign == "largest_magnitude":
        largest = jnp.take_along_axis(
            stacked,
            jnp.argmax(jnp.abs(stacked), axis=0, keepdims=True),
            axis=0,
        )[0]
        consensus = jnp.where(consensus == 0, jnp.sign(largest), consensus)
    elif zero_sign != "zero":
        raise ValueError("TIES zero_sign must be 'zero', 'positive', or 'largest_magnitude'.")
    agree = jnp.logical_and(stacked != 0, jnp.sign(stacked) == consensus[None, ...])
    numerator = jnp.sum(jnp.where(agree, stacked, 0), axis=0)
    denominator = jnp.sum(agree.astype(jnp.float32), axis=0)
    merged = jnp.where(denominator > 0, numerator / denominator, 0)
    return merged * final_scale


def _dare_leaf(leaf, *, drop_rate: float, key: jax.Array, rescale: bool):
    if not eqx.is_inexact_array(leaf):
        return leaf
    keep_prob = 1.0 - drop_rate
    mask = jax.random.bernoulli(key, keep_prob, leaf.shape)
    value = jnp.where(mask, leaf, 0)
    return value / keep_prob if rescale and keep_prob > 0 else value


def _dare_delta_global(
    delta: PyTree,
    *,
    drop_rate: float,
    key: jax.Array,
    rescale: bool,
) -> PyTree:
    keep_prob = 1.0 - drop_rate
    leaves, treedef = jtu.tree_flatten(delta)
    total_size = sum(leaf.size for leaf in leaves if eqx.is_inexact_array(leaf))
    if total_size == 0:
        return delta
    flat_mask = jax.random.bernoulli(key, keep_prob, (total_size,))
    offset = 0
    masked = []
    for leaf in leaves:
        if not eqx.is_inexact_array(leaf):
            masked.append(leaf)
            continue
        size = leaf.size
        mask = flat_mask[offset : offset + size].reshape(leaf.shape)
        value = jnp.where(mask, leaf, 0)
        masked.append(value / keep_prob if rescale and keep_prob > 0 else value)
        offset += size
    return jtu.tree_unflatten(treedef, masked)


def _breadcrumbs_leaf(leaf, *, bottom_fraction: float, top_fraction: float):
    if leaf.size == 0:
        return leaf
    flat_abs = jnp.abs(leaf).reshape(-1)
    low = jnp.quantile(flat_abs, bottom_fraction)
    high = jnp.quantile(flat_abs, 1.0 - top_fraction)
    return jnp.where((jnp.abs(leaf) >= low) & (jnp.abs(leaf) <= high), leaf, 0)


def _validate_breadcrumbs_fractions(
    bottom_fraction: float,
    top_fraction: float,
) -> None:
    if not 0.0 <= bottom_fraction < 1.0:
        raise ValueError("Breadcrumbs bottom_fraction must satisfy 0 <= value < 1.")
    if not 0.0 <= top_fraction < 1.0:
        raise ValueError("Breadcrumbs top_fraction must satisfy 0 <= value < 1.")
    if bottom_fraction + top_fraction >= 1.0:
        raise ValueError(
            "Breadcrumbs bottom_fraction + top_fraction must be less than 1."
        )


def _normalize_fisher_tree(fisher: PyTree) -> PyTree:
    leaves, treedef = jtu.tree_flatten(fisher)
    norm_parts = [
        jnp.sum(jnp.square(leaf))
        for leaf in leaves
        if eqx.is_inexact_array(leaf) and leaf.size > 0
    ]
    if not norm_parts:
        return fisher
    norm_sq = norm_parts[0]
    for part in norm_parts[1:]:
        norm_sq = norm_sq + part
    norm = jnp.sqrt(norm_sq)
    safe_norm = jnp.where(norm > 0, norm, 1.0)
    normalized = [
        jnp.where(norm > 0, leaf / safe_norm, leaf)
        if eqx.is_inexact_array(leaf)
        else leaf
        for leaf in leaves
    ]
    return jtu.tree_unflatten(treedef, normalized)


def _fisher_leaf(model_leaves, fisher_leaves, *, eps: float):
    first = model_leaves[0]
    if not eqx.is_inexact_array(first):
        return first
    numerator = jnp.zeros_like(first)
    denominator = jnp.zeros_like(first)
    for model_leaf, fisher_leaf in zip(model_leaves, fisher_leaves, strict=True):
        numerator = numerator + model_leaf * fisher_leaf
        denominator = denominator + fisher_leaf
    return numerator / jnp.maximum(denominator, eps)


def _regmean_leaf(
    model_leaves,
    covariance_leaves,
    *,
    ridge: float,
    non_matrix_policy: str,
    solver: str,
):
    first = model_leaves[0]
    if not eqx.is_inexact_array(first):
        return first
    _check_leaf_shapes(model_leaves)
    if first.ndim == 2 and _all_input_covariances(first, covariance_leaves):
        covariance_sum = jnp.zeros_like(covariance_leaves[0])
        weighted_sum = jnp.zeros_like(first)
        for model_leaf, covariance in zip(model_leaves, covariance_leaves, strict=True):
            covariance_sum = covariance_sum + covariance
            weighted_sum = weighted_sum + model_leaf @ covariance
        eye = jnp.eye(covariance_sum.shape[0], dtype=covariance_sum.dtype)
        system = covariance_sum + ridge * eye
        return _solve_regmean_system(system, weighted_sum, solver=solver)
    if first.ndim == 2 and non_matrix_policy == "error":
        raise ValueError("RegMean requires covariance leaves shaped (input_dim, input_dim).")
    return _non_matrix_leaf(model_leaves, non_matrix_policy)


def _solve_regmean_system(system, weighted_sum, *, solver: str):
    rhs = weighted_sum.T
    if solver == "solve":
        return jnp.linalg.solve(system.T, rhs).T
    if solver == "cholesky":
        factor = jnp.linalg.cholesky(system)
        intermediate = jnp.linalg.solve(factor, rhs)
        return jnp.linalg.solve(factor.T, intermediate).T
    if solver == "svd":
        left, singular_values, right = jnp.linalg.svd(system, full_matrices=False)
        cutoff = (
            jnp.finfo(system.dtype).eps
            * max(system.shape)
            * jnp.max(singular_values)
        )
        inv_singular_values = jnp.where(
            singular_values > cutoff,
            1.0 / singular_values,
            0.0,
        )
        return ((right.T * inv_singular_values) @ left.T @ rhs).T
    raise ValueError("RegMean solver must be 'cholesky', 'solve', or 'svd'.")


def _all_input_covariances(first, covariance_leaves) -> bool:
    input_dim = first.shape[1]
    return all(
        eqx.is_inexact_array(covariance)
        and covariance.shape == (input_dim, input_dim)
        for covariance in covariance_leaves
    )


def _check_same_architecture(
    models: Sequence[PyTree],
    *,
    include_head: bool,
) -> None:
    hashes = tuple(_architecture_hash(model, include_head=include_head) for model in models)
    if len(set(hashes)) > 1:
        raise ValueError("Models must have the same architecture for this merge operation.")


def _check_same_base_metadata(
    models: Sequence[PyTree],
    *,
    require_same_base_hash: bool,
) -> None:
    if not require_same_base_hash:
        return
    hashes = tuple(_base_hash_metadata(model) for model in models)
    present = tuple(hash_value for hash_value in hashes if hash_value)
    if not present:
        return
    if len(present) != len(models):
        raise ValueError(
            "Same-base compatibility requires base checkpoint metadata on every model."
        )
    if len(set(present)) > 1:
        raise ValueError("Models must declare the same base checkpoint hash.")


def _check_task_vector_base(base_model: PyTree, vector: TaskVector) -> None:
    if vector.base_architecture_hash:
        actual_architecture = _architecture_hash(
            base_model,
            include_head=vector.include_head,
        )
        if actual_architecture != vector.base_architecture_hash:
            raise ValueError(
                "Task vector architecture mismatch: expected "
                f"{vector.base_architecture_hash}, got {actual_architecture}."
            )
    if vector.base_checkpoint_hash:
        actual_checkpoint = _checkpoint_hash(base_model, include_head=vector.include_head)
        if actual_checkpoint != vector.base_checkpoint_hash:
            raise ValueError(
                "Task vector base checkpoint mismatch: expected "
                f"{vector.base_checkpoint_hash}, got {actual_checkpoint}."
            )
    if vector.logical_id_table_hash:
        actual_logical_ids = _logical_id_table_hash(
            base_model,
            include_head=vector.include_head,
        )
        if actual_logical_ids != vector.logical_id_table_hash:
            raise ValueError(
                "Task vector logical-ID table mismatch: expected "
                f"{vector.logical_id_table_hash}, got {actual_logical_ids}."
            )


def _check_vector_compatibility(
    vectors: Sequence[TaskVector],
    *,
    require_same_base_hash: bool,
) -> None:
    architecture_hashes = {
        vector.base_architecture_hash
        for vector in vectors
        if vector.base_architecture_hash
    }
    if len(architecture_hashes) > 1:
        raise ValueError("Task vectors must share the same base architecture.")
    logical_id_hashes = tuple(
        vector.logical_id_table_hash
        for vector in vectors
        if vector.logical_id_table_hash
    )
    if logical_id_hashes and len(logical_id_hashes) != len(vectors):
        raise ValueError(
            "Task vectors with logical-ID table metadata must not be merged "
            "with vectors missing that metadata."
        )
    if len(set(logical_id_hashes)) > 1:
        raise ValueError("Task vectors must share the same logical-ID table.")
    if not require_same_base_hash:
        return
    checkpoint_hashes = {
        vector.base_checkpoint_hash
        for vector in vectors
        if vector.base_checkpoint_hash
    }
    if len(checkpoint_hashes) > 1:
        raise ValueError("Task vectors must share the same base checkpoint.")


def _architecture_hash(model: PyTree, *, include_head: bool) -> str:
    digest = hashlib.sha256()
    for key_path, leaf in _iter_hashable_leaves(model, include_head=include_head):
        array = jnp.asarray(leaf)
        digest.update(path_to_str(key_path_to_path(key_path)).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(str(array.dtype).encode())
    return digest.hexdigest()


def _logical_id_table_hash(model: PyTree, *, include_head: bool) -> str:
    digest = hashlib.sha256()
    for key_path, _leaf in _iter_hashable_leaves(model, include_head=include_head):
        digest.update(path_to_str(key_path_to_path(key_path)).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _checkpoint_hash(model: PyTree, *, include_head: bool) -> str:
    digest = hashlib.sha256()
    for key_path, leaf in _iter_hashable_leaves(model, include_head=include_head):
        array = np.asarray(leaf)
        digest.update(path_to_str(key_path_to_path(key_path)).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _iter_hashable_leaves(model: PyTree, *, include_head: bool):
    filtered = eqx.filter(model, eqx.is_inexact_array)
    for key_path, leaf in jtu.tree_leaves_with_path(filtered):
        path = key_path_to_path(key_path)
        if not eqx.is_inexact_array(leaf):
            continue
        if not include_head and _is_head_path(path):
            continue
        yield key_path, leaf


def _is_head_path(path) -> bool:
    return any(str(part) in {"head", "classifier"} for part in path)


def _base_hash_metadata(model: PyTree) -> str | None:
    for name in ("base_checkpoint_id", "base_checkpoint_hash", "base_model_hash"):
        value = getattr(model, name, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(model, Mapping):
        for name in ("base_checkpoint_id", "base_checkpoint_hash", "base_model_hash"):
            value = model.get(name)
            if isinstance(value, str) and value:
                return value
        metadata = model.get("metadata")
        if isinstance(metadata, Mapping):
            for name in ("base_checkpoint_id", "base_checkpoint_hash", "base_model_hash"):
                value = metadata.get(name)
                if isinstance(value, str) and value:
                    return value
    metadata = getattr(model, "metadata", None)
    if isinstance(metadata, dict):
        for name in ("base_checkpoint_id", "base_checkpoint_hash", "base_model_hash"):
            value = metadata.get(name)
            if isinstance(value, str) and value:
                return value
    return None


__all__ = (
    "TaskVector",
    "TaskVectorConfig",
    "KnOTSConfig",
    "KnOTSMerging",
    "MergeMethod",
    "MergePlan",
    "TIESConfig",
    "UniformSoupConfig",
    "WiSEFTConfig",
    "BreadcrumbsConfig",
    "DARETransform",
    "FisherMergeConfig",
    "GreedySoupConfig",
    "RegMeanConfig",
    "apply_task_vector",
    "breadcrumbs_task_vector",
    "dare_task_vector",
    "fisher_merge",
    "greedy_soup",
    "interpolate_models",
    "knots_task_vector",
    "regmean_merge",
    "task_vector_bundle",
    "task_vector",
    "ties_merge",
    "uniform_soup",
)
