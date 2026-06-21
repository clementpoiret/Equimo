"""Pure regularization helpers for fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from ._typing import PyTree


@dataclass(frozen=True)
class L2SPConfig:
    """L2-SP coefficient metadata."""

    coefficient_hint: float = 1e-3


@dataclass(frozen=True)
class FeatureDistillationConfig:
    """Feature distillation metadata."""

    layers: tuple[str, ...] = ("50%", "100%")
    metric: str = "mse"


def l2_sp_loss(model: PyTree, reference: PyTree, *, coefficient: float = 1.0) -> jax.Array:
    """Return L2-SP loss between compatible inexact array leaves."""

    leaves = jtu.tree_leaves(
        jtu.tree_map(
            lambda leaf, ref: jnp.sum((leaf - ref) ** 2)
            if eqx.is_inexact_array(leaf) and eqx.is_inexact_array(ref)
            else jnp.asarray(0.0),
            model,
            reference,
        )
    )
    return jnp.sum(jnp.stack(leaves)) * coefficient


def feature_distillation_loss(
    student_features: jax.Array,
    teacher_features: jax.Array,
    *,
    metric: str = "mse",
) -> jax.Array:
    """Return feature distillation loss with stopped teacher gradients."""

    teacher_features = jax.lax.stop_gradient(teacher_features)
    if metric == "mse":
        return jnp.mean((student_features - teacher_features) ** 2)
    if metric == "cosine":
        student = _normalize(student_features)
        teacher = _normalize(teacher_features)
        return 1.0 - jnp.mean(jnp.sum(student * teacher, axis=-1))
    raise ValueError(f"Unsupported feature distillation metric {metric!r}.")


def _normalize(x: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, 1e-12)


__all__ = (
    "FeatureDistillationConfig",
    "L2SPConfig",
    "feature_distillation_loss",
    "l2_sp_loss",
)
