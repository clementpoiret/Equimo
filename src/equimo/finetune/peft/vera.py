"""VeRA modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .._typing import Path, PyTree
from ..config import TargetSpec
from ..paths import path_to_str
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path


@dataclass(frozen=True)
class VeRAConfig:
    """Configuration for VeRA linear wrappers."""

    rank: int = 256
    shared: bool = True
    trainable_output_scale_init: float = 0.0
    target: TargetSpec = field(
        default_factory=lambda: TargetSpec(tags=("attention.proj", "mlp.fc2"))
    )


class VeRALinear(eqx.Module):
    """VeRA wrapper with frozen random bases and trainable scales."""

    base: eqx.nn.Linear
    vera_A: jax.Array
    vera_B: jax.Array
    vera_input_scale: jax.Array
    vera_output_scale: jax.Array

    def __init__(
        self,
        base: eqx.nn.Linear,
        *,
        rank: int,
        key: jax.Array,
        output_scale_init: float = 0.0,
        vera_A: jax.Array | None = None,
        vera_B: jax.Array | None = None,
        vera_input_scale: jax.Array | None = None,
        vera_output_scale: jax.Array | None = None,
    ):
        key_a, key_b = jr.split(key, 2)
        self.base = base
        self.vera_A = (
            jr.normal(key_a, (rank, base.in_features), dtype=base.weight.dtype)
            / jnp.sqrt(base.in_features)
            if vera_A is None
            else vera_A
        )
        self.vera_B = (
            jr.normal(key_b, (base.out_features, rank), dtype=base.weight.dtype)
            / jnp.sqrt(rank)
            if vera_B is None
            else vera_B
        )
        self.vera_input_scale = (
            jnp.ones((rank,), dtype=base.weight.dtype)
            if vera_input_scale is None
            else vera_input_scale
        )
        self.vera_output_scale = (
            jnp.full((base.out_features,), output_scale_init, dtype=base.weight.dtype)
            if vera_output_scale is None
            else vera_output_scale
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        vera_A = jax.lax.stop_gradient(self.vera_A)
        vera_B = jax.lax.stop_gradient(self.vera_B)
        hidden = (vera_A @ x) * self.vera_input_scale
        update = (vera_B @ hidden) * self.vera_output_scale
        return self.base(x) + update


def apply_vera(
    model: PyTree,
    config: VeRAConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply VeRA wrappers to selected linear modules."""

    config = VeRAConfig() if config is None else config
    module_paths = _target_linear_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(module_paths))
    updated = model
    for module_path, subkey in zip(module_paths, keys, strict=True):
        module = get_path(updated, module_path)
        if isinstance(module, VeRALinear):
            continue
        if not isinstance(module, eqx.nn.Linear):
            raise TypeError(
                f"VeRA target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected eqx.nn.Linear."
            )
        updated = eqx.tree_at(
            lambda tree, p=module_path: get_path(tree, p),
            updated,
            VeRALinear(
                module,
                rank=config.rank,
                key=subkey,
                output_scale_init=config.trainable_output_scale_init,
            ),
        )
    return updated


def _target_linear_paths(model: PyTree, target: TargetSpec, *, tagger: Tagger) -> tuple[Path, ...]:
    paths = {
        info.path[:-1]
        for info in resolve_target(model, target, tagger=tagger)
        if info.path[-1:] in (("weight",), ("bias",))
    }
    return tuple(sorted(paths, key=path_to_str))


__all__ = (
    "VeRAConfig",
    "VeRALinear",
    "apply_vera",
)
