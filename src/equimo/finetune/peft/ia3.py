"""IA3 activation scaling wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .._typing import Path, PyTree
from ..config import TargetSpec
from ..paths import path_to_str
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path


@dataclass(frozen=True)
class IA3Config:
    """Configuration for IA3 activation scaling."""

    target: TargetSpec = TargetSpec(tags=("attention.proj", "mlp.fc2"))
    init: float = 1.0


class IA3Linear(eqx.Module):
    """Wrap a linear module with trainable output-channel IA3 scaling."""

    base: eqx.nn.Linear
    ia3: jax.Array

    def __init__(
        self,
        base: eqx.nn.Linear,
        *,
        init: float = 1.0,
        ia3: jax.Array | None = None,
    ):
        self.base = base
        self.ia3 = (
            jnp.full((base.out_features,), init, dtype=base.weight.dtype)
            if ia3 is None
            else ia3
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.base(x) * self.ia3


def apply_ia3(
    model: PyTree,
    config: IA3Config | None = None,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply IA3 wrappers to selected linear modules."""

    config = IA3Config() if config is None else config
    module_paths = _target_linear_paths(model, config.target, tagger=tagger)
    updated = model
    for module_path in module_paths:
        module = get_path(updated, module_path)
        if isinstance(module, IA3Linear):
            continue
        if not isinstance(module, eqx.nn.Linear):
            raise TypeError(
                f"IA3 target {path_to_str(module_path)!r} is "
                f"{type(module).__name__}, expected eqx.nn.Linear."
            )
        updated = eqx.tree_at(
            lambda tree, p=module_path: get_path(tree, p),
            updated,
            IA3Linear(module, init=config.init),
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
    "IA3Config",
    "IA3Linear",
    "apply_ia3",
)
