"""Model merging and task-vector utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from ._typing import PyTree
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
    if config is not None:
        alpha = config.alpha
        include_head = config.include_head
        strict_shapes = config.strict_shapes
        if config.require_same_architecture_hash:
            _check_same_architecture(
                (base_model, tuned_model),
                include_head=include_head,
            )

    return jtu.tree_map_with_path(
        lambda path, base, tuned: _interpolate_leaf(
            key_path_to_path(path),
            base,
            tuned,
            alpha=alpha,
            include_head=include_head,
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
    if config is not None:
        density = config.density
        final_scale = config.final_scale
        if config.density_scope != "per_tensor":
            raise ValueError("TIESConfig.density_scope currently supports 'per_tensor'.")
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
    merged = _merge_deltas(
        [vector.delta for vector in vectors],
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
    )


def dare_task_vector(
    vector: TaskVector,
    *,
    drop_rate: float = 0.90,
    key: jax.Array,
    rescale: bool = True,
    config: DARETransform | None = None,
) -> TaskVector:
    """Apply a static DARE mask to a task vector."""

    if config is not None:
        drop_rate = config.drop_rate
        rescale = config.rescale
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
    )


def fisher_merge(
    models: Sequence[PyTree],
    fishers: Sequence[PyTree] | None = None,
    *,
    eps: float = 1e-8,
    config: FisherMergeConfig | None = None,
) -> PyTree:
    """Fisher-weighted merge requiring external Fisher statistics."""

    if config is not None:
        eps = config.eps
        if config.fisher != "diagonal":
            raise ValueError("fisher_merge currently supports fisher='diagonal'.")
    if fishers is None:
        raise ValueError("fisher_merge requires external Fisher statistics.")
    if len(models) != len(fishers):
        raise ValueError("models and fishers must have the same length.")
    first_model, *other_models = models
    first_fisher, *other_fishers = fishers
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

    if config is not None:
        ridge = config.ridge
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
        ),
        first_model,
        *other_models,
        first_covariance,
        *other_covariances,
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


def _interpolate_leaf(
    path,
    base,
    tuned,
    *,
    alpha: float,
    include_head: bool,
    strict_shapes: bool,
):
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


def _breadcrumbs_leaf(leaf, *, bottom_fraction: float, top_fraction: float):
    flat_abs = jnp.abs(leaf).reshape(-1)
    low = jnp.quantile(flat_abs, bottom_fraction)
    high = jnp.quantile(flat_abs, 1.0 - top_fraction)
    return jnp.where((jnp.abs(leaf) >= low) & (jnp.abs(leaf) <= high), leaf, 0)


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


def _regmean_leaf(model_leaves, covariance_leaves, *, ridge: float):
    first = model_leaves[0]
    if not eqx.is_inexact_array(first):
        return first
    if first.ndim == 2 and _all_input_covariances(first, covariance_leaves):
        covariance_sum = jnp.zeros_like(covariance_leaves[0])
        weighted_sum = jnp.zeros_like(first)
        for model_leaf, covariance in zip(model_leaves, covariance_leaves, strict=True):
            covariance_sum = covariance_sum + covariance
            weighted_sum = weighted_sum + model_leaf @ covariance
        eye = jnp.eye(covariance_sum.shape[0], dtype=covariance_sum.dtype)
        system = covariance_sum + ridge * eye
        return jnp.linalg.solve(system.T, weighted_sum.T).T
    if all(
        eqx.is_inexact_array(covariance) and covariance.shape == first.shape
        for covariance in covariance_leaves
    ):
        return _fisher_leaf(model_leaves, covariance_leaves, eps=max(ridge, 1e-12))
    return _soup_leaf(
        (),
        model_leaves,
        _resolve_weights(len(model_leaves), "equal"),
        include_head=True,
    )


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
    "regmean_merge",
    "task_vector",
    "ties_merge",
    "uniform_soup",
)
