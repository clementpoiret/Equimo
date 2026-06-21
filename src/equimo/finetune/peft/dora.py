"""DoRA modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import TargetSpec
from ..paths import key_path_to_path, path_to_str
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path
from .lora import ScalingMode


@dataclass(frozen=True)
class DoRAConfig:
    """Configuration for DoRA linear wrappers."""

    rank: int = 8
    alpha: float = 16.0
    scaling: ScalingMode = "alpha_over_r"
    dropout: float = 0.0
    init: str = "kaiming_A_zero_B"
    magnitude_axis: Literal["logical_output"] = "logical_output"
    magnitude_init: Literal["base_weight_norm"] = "base_weight_norm"
    norm_gradient: Literal["full", "detached"] = "full"
    norm_impl: Literal["dense", "factored"] = "dense"
    eps: float = 1e-6
    train_base: bool = False
    mergeable: bool = True
    target: TargetSpec = field(
        default_factory=lambda: TargetSpec(
            tags_any=("attention.qkv", "attention.proj"),
        )
    )


@dataclass(frozen=True)
class DoRARecipe:
    """Recipe metadata for DoRA fine-tuning."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target: tuple[str, ...] = ("attention.qkv", "attention.proj")
    external_lr_hint: str = "slightly_lower_than_lora"

    def to_config(self) -> DoRAConfig:
        """Convert recipe metadata to a DoRA module config."""

        return DoRAConfig(
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target=TargetSpec(tags_any=self.target),
        )


class DoRALinear(eqx.Module):
    """Weight-decomposed low-rank adaptation for linear modules."""

    base: eqx.nn.Linear
    lora_A: jax.Array
    lora_B: jax.Array
    magnitude: jax.Array
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    scaling_mode: ScalingMode = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    train_base: bool = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)
    norm_gradient: Literal["full", "detached"] = eqx.field(static=True)
    norm_impl: Literal["dense", "factored"] = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.nn.Linear,
        *,
        rank: int,
        alpha: float,
        scaling: ScalingMode,
        dropout: float,
        eps: float,
        train_base: bool,
        mergeable: bool,
        key: jax.Array,
        magnitude_init: str = "base_weight_norm",
        norm_gradient: Literal["full", "detached"] = "full",
        norm_impl: Literal["dense", "factored"] = "dense",
        lora_A: jax.Array | None = None,
        lora_B: jax.Array | None = None,
        magnitude: jax.Array | None = None,
    ):
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling_mode = scaling
        self.dropout = dropout
        self.eps = eps
        self.train_base = train_base
        self.mergeable = mergeable
        self.norm_gradient = norm_gradient
        self.norm_impl = norm_impl
        if lora_A is None or lora_B is None:
            lora_A, lora_B = _init_lora(base, rank, key)
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.magnitude = (
            _init_magnitude(base, magnitude_init)
            if magnitude is None
            else magnitude
        )

    @property
    def scaling(self) -> float:
        if self.scaling_mode == "alpha_over_r":
            return float(self.alpha / self.rank)
        if self.scaling_mode == "alpha_over_sqrt_r":
            return float(self.alpha / jnp.sqrt(self.rank))
        raise ValueError(f"Unsupported DoRA scaling mode {self.scaling_mode!r}.")

    def weight(self) -> jax.Array:
        direction = self.base.weight + self._delta_weight()
        direction_norm = jnp.linalg.norm(direction, axis=1, keepdims=True)
        if self.norm_gradient == "detached":
            direction_norm = jax.lax.stop_gradient(direction_norm)
        direction = direction / jnp.maximum(direction_norm, self.eps)
        return direction * self.magnitude[:, None]

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        if self.dropout > 0.0 and not inference:
            if key is None:
                raise ValueError("A PRNG key is required when DoRA dropout is active.")
            y = self._dropout_weight_projection(x, key)
        else:
            y = self.weight() @ x
        if self.base.bias is not None:
            y = y + self.base.bias
        return y

    def _delta_weight(self) -> jax.Array:
        return (self.lora_B @ self.lora_A) * self.scaling

    def _dropout_weight_projection(self, x: jax.Array, key: jax.Array) -> jax.Array:
        dense_direction = self.base.weight + self._delta_weight()
        direction_norm = jnp.linalg.norm(dense_direction, axis=1)
        if self.norm_gradient == "detached":
            direction_norm = jax.lax.stop_gradient(direction_norm)
        x_drop = _dropout(x, self.dropout, key)
        projected = (
            self.base.weight @ x
            + (self.lora_B @ (self.lora_A @ x_drop)) * self.scaling
        )
        return projected * self.magnitude / jnp.maximum(direction_norm, self.eps)


class DoRAMergedLinear(DoRALinear):
    """DoRA wrapper for fused projections such as QKV linears."""


def apply_dora(
    model: PyTree,
    config: DoRAConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply DoRA wrappers to selected linears."""

    config = DoRAConfig() if config is None else config
    module_paths = _target_linear_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, DoRALinear):
            continue
        if not isinstance(module, eqx.nn.Linear):
            raise TypeError(
                f"DoRA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected eqx.nn.Linear."
            )
        wrapper_type = DoRAMergedLinear if "qkv" in {str(part) for part in module_path} else DoRALinear
        if config.init != "kaiming_A_zero_B":
            raise ValueError(
                f"Unsupported DoRA init {config.init!r}; expected kaiming_A_zero_B."
            )
        updated = eqx.tree_at(
            lambda tree, p=module_path: get_path(tree, p),
            updated,
            wrapper_type(
                module,
                rank=config.rank,
                alpha=config.alpha,
                scaling=config.scaling,
                dropout=config.dropout,
                eps=config.eps,
                train_base=config.train_base,
                mergeable=config.mergeable,
                magnitude_init=config.magnitude_init,
                norm_gradient=config.norm_gradient,
                norm_impl=config.norm_impl,
                key=subkey,
            ),
        )
    return updated


def merge_dora(model: PyTree) -> PyTree:
    """Fold every DoRA wrapper into its base ``eqx.nn.Linear`` module."""

    updated = model
    for path, module in iter_dora_modules(updated):
        if not module.mergeable:
            raise ValueError("This DoRA module is not mergeable.")
        base = eqx.tree_at(
            lambda linear: linear.weight,
            module.base,
            module.weight().astype(module.base.weight.dtype),
        )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, base)
    return updated


def iter_dora_modules(model: PyTree) -> tuple[tuple[Path, DoRALinear], ...]:
    """Return path/module pairs for DoRA wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, DoRALinear),
        )
        if isinstance(leaf, DoRALinear)
    )


def _target_linear_paths(model: PyTree, target: TargetSpec, *, tagger: Tagger) -> tuple[Path, ...]:
    paths = {
        info.path[:-1]
        for info in resolve_target(model, target, tagger=tagger)
        if info.path[-1:] in (("weight",), ("bias",))
    }
    return tuple(sorted(paths, key=path_to_str))


def _init_lora(
    base: eqx.nn.Linear,
    rank: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    key_a, _ = jr.split(key, 2)
    bound = jnp.sqrt(6.0 / base.in_features)
    lora_A = jr.uniform(
        key_a,
        (rank, base.in_features),
        minval=-bound,
        maxval=bound,
        dtype=base.weight.dtype,
    )
    lora_B = jnp.zeros((base.out_features, rank), dtype=base.weight.dtype)
    return lora_A, lora_B


def _init_magnitude(base: eqx.nn.Linear, magnitude_init: str) -> jax.Array:
    if magnitude_init == "base_weight_norm":
        return jnp.linalg.norm(base.weight, axis=1).astype(base.weight.dtype)
    raise ValueError(
        "Unsupported DoRA magnitude_init "
        f"{magnitude_init!r}; expected 'base_weight_norm'."
    )


def _dropout(x: jax.Array, rate: float, key: jax.Array) -> jax.Array:
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


__all__ = (
    "DoRAConfig",
    "DoRALinear",
    "DoRAMergedLinear",
    "DoRARecipe",
    "apply_dora",
    "iter_dora_modules",
    "merge_dora",
)
