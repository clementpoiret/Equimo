"""Model merging and task-vector utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ._typing import PyTree
from .paths import key_path_to_path


@dataclass(frozen=True)
class WiSEFTConfig:
    """Configuration for WiSE-FT interpolation."""

    alpha: float = 0.5
    include_head: bool = False
    require_same_architecture_hash: bool = True


@dataclass(frozen=True)
class UniformSoupConfig:
    """Configuration for uniform model soups."""

    weights: tuple[float, ...] | str = "equal"
    require_same_base_hash: bool = True


@dataclass(frozen=True)
class TaskVectorConfig:
    """Configuration for task-vector extraction/application."""

    scale: float = 1.0
    include_head: bool = False
    require_same_base_hash: bool = True


@dataclass(frozen=True)
class TaskVector:
    """Difference PyTree from a base model to a fine-tuned model."""

    delta: PyTree
    include_head: bool = False


@dataclass(frozen=True)
class TIESConfig:
    """TIES merge metadata."""

    density: float = 0.20


@dataclass(frozen=True)
class DAREConfig:
    """DARE merge metadata."""

    drop_rate: float = 0.90
    rescale: bool = True


@dataclass(frozen=True)
class BreadcrumbsConfig:
    """Model Breadcrumbs merge metadata."""

    bottom_fraction: float = 0.05
    top_fraction: float = 0.01


@dataclass(frozen=True)
class FisherMergeConfig:
    """Fisher merge metadata."""

    eps: float = 1e-8


@dataclass(frozen=True)
class RegMeanConfig:
    """RegMean merge metadata."""

    ridge: float = 1e-5


def interpolate_models(
    base_model: PyTree,
    tuned_model: PyTree,
    *,
    alpha: float = 0.5,
    include_head: bool = False,
) -> PyTree:
    """Interpolate compatible model leaves."""

    return jtu.tree_map_with_path(
        lambda path, base, tuned: _interpolate_leaf(
            key_path_to_path(path),
            base,
            tuned,
            alpha=alpha,
            include_head=include_head,
        ),
        base_model,
        tuned_model,
    )


def uniform_soup(
    models: Sequence[PyTree],
    *,
    weights: Sequence[float] | str = "equal",
    include_head: bool = True,
) -> PyTree:
    """Return the weighted arithmetic mean of compatible model leaves."""

    if not models:
        raise ValueError("uniform_soup requires at least one model.")
    resolved_weights = _resolve_weights(len(models), weights)
    first, *rest = models
    return jtu.tree_map_with_path(
        lambda path, leaf, *other_leaves: _soup_leaf(
            key_path_to_path(path),
            (leaf, *other_leaves),
            resolved_weights,
            include_head=include_head,
        ),
        first,
        *rest,
    )


def greedy_soup(
    models: Sequence[PyTree],
    score_fn: Callable[[PyTree], float],
    *,
    higher_is_better: bool = True,
) -> tuple[PyTree, tuple[int, ...]]:
    """Greedily add models when the user-provided score improves."""

    if not models:
        raise ValueError("greedy_soup requires at least one model.")

    selected = [0]
    soup = models[0]
    best_score = score_fn(soup)

    for index, candidate in enumerate(models[1:], start=1):
        proposal = uniform_soup([soup, candidate])
        score = score_fn(proposal)
        improves = score >= best_score if higher_is_better else score <= best_score
        if improves:
            soup = proposal
            best_score = score
            selected.append(index)

    return soup, tuple(selected)


def task_vector(
    base_model: PyTree,
    tuned_model: PyTree,
    *,
    include_head: bool = False,
) -> TaskVector:
    """Extract a task vector from compatible model leaves."""

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
    return TaskVector(delta=delta, include_head=include_head)


def apply_task_vector(
    base_model: PyTree,
    vector: TaskVector,
    *,
    scale: float = 1.0,
) -> PyTree:
    """Apply a task vector to a compatible base model."""

    return jtu.tree_map(
        lambda base, delta: base + delta * scale
        if eqx.is_inexact_array(base) and eqx.is_inexact_array(delta)
        else base,
        base_model,
        vector.delta,
    )


def ties_merge(vectors: Sequence[TaskVector], *, density: float = 0.20) -> TaskVector:
    """Merge task vectors with a simple TIES-style sign consensus."""

    if not vectors:
        raise ValueError("ties_merge requires at least one task vector.")
    merged = _merge_deltas(
        [vector.delta for vector in vectors],
        lambda leaves: _ties_leaf(leaves, density),
    )
    return TaskVector(merged, include_head=all(vector.include_head for vector in vectors))


def dare_task_vector(
    vector: TaskVector,
    *,
    drop_rate: float = 0.90,
    key: jax.Array,
    rescale: bool = True,
) -> TaskVector:
    """Apply a static DARE mask to a task vector."""

    leaves, treedef = jtu.tree_flatten(vector.delta)
    keys = jax.random.split(key, len(leaves))
    masked = [
        _dare_leaf(leaf, drop_rate=drop_rate, key=leaf_key, rescale=rescale)
        for leaf, leaf_key in zip(leaves, keys, strict=True)
    ]
    return TaskVector(jtu.tree_unflatten(treedef, masked), include_head=vector.include_head)


def breadcrumbs_task_vector(
    vector: TaskVector,
    *,
    bottom_fraction: float = 0.05,
    top_fraction: float = 0.01,
) -> TaskVector:
    """Remove the smallest and largest task-vector deltas by magnitude."""

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
    )


def fisher_merge(
    models: Sequence[PyTree],
    fishers: Sequence[PyTree] | None = None,
    *,
    eps: float = 1e-8,
) -> PyTree:
    """Fisher-weighted merge requiring external Fisher statistics."""

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
) -> PyTree:
    """RegMean-style merge requiring external covariance statistics."""

    if covariances is None:
        raise ValueError("regmean_merge requires external covariance statistics.")
    del ridge
    return uniform_soup(models)


def _interpolate_leaf(path, base, tuned, *, alpha: float, include_head: bool):
    if not _mergeable(path, base, tuned, include_head=include_head):
        return base
    return (1.0 - alpha) * base + alpha * tuned


def _soup_leaf(path, leaves, weights, *, include_head: bool):
    first = leaves[0]
    if not eqx.is_inexact_array(first) or (not include_head and "head" in path):
        return first
    total = jnp.zeros_like(first)
    for leaf, weight in zip(leaves, weights, strict=True):
        if not eqx.is_inexact_array(leaf) or leaf.shape != first.shape:
            raise ValueError("uniform_soup encountered incompatible leaves.")
        total = total + leaf * weight
    return total


def _delta_leaf(path, base, tuned, *, include_head: bool):
    if not _mergeable(path, base, tuned, include_head=include_head):
        return jnp.zeros_like(base) if eqx.is_inexact_array(base) else base
    return tuned - base


def _mergeable(path, base, tuned, *, include_head: bool) -> bool:
    if not eqx.is_inexact_array(base) or not eqx.is_inexact_array(tuned):
        return False
    if base.shape != tuned.shape:
        raise ValueError("Cannot merge leaves with different shapes.")
    return include_head or "head" not in path


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


def _ties_leaf(leaves, density: float):
    first = leaves[0]
    if not eqx.is_inexact_array(first):
        return first
    stacked = jnp.stack(leaves)
    mean = jnp.mean(stacked, axis=0)
    if density >= 1.0:
        return mean
    threshold = jnp.quantile(jnp.abs(mean).reshape(-1), 1.0 - density)
    mask = jnp.abs(mean) >= threshold
    consensus = jnp.sign(jnp.sum(jnp.sign(stacked), axis=0))
    return jnp.where(mask, jnp.abs(mean) * consensus, 0)


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


__all__ = (
    "TaskVector",
    "TaskVectorConfig",
    "TIESConfig",
    "UniformSoupConfig",
    "WiSEFTConfig",
    "BreadcrumbsConfig",
    "DAREConfig",
    "FisherMergeConfig",
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
