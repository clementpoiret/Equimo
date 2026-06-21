"""IA3 activation scaling wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import TargetSpec
from ..paths import key_path_to_path, path_to_str
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path


@dataclass(frozen=True)
class IA3Config:
    """Configuration for IA3 activation scaling."""

    target: TargetSpec = TargetSpec(tags_any=("attention.k", "attention.v", "mlp.hidden"))
    init: float = 1.0
    axis: Literal["feature"] = "feature"
    train_head: bool = True
    mergeable: bool = True


class IA3Linear(eqx.Module):
    """Wrap a linear module with trainable output-channel IA3 scaling."""

    base: eqx.nn.Linear
    ia3: jax.Array
    mergeable: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.nn.Linear,
        *,
        init: float = 1.0,
        ia3: jax.Array | None = None,
        mergeable: bool = True,
    ):
        self.base = base
        self.mergeable = mergeable
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
    if config.axis != "feature":
        raise ValueError("IA3Config.axis currently supports only 'feature'.")
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
            IA3Linear(module, init=config.init, mergeable=config.mergeable),
        )
    return updated


def merge_ia3(model: PyTree) -> PyTree:
    """Fold IA3 output scales into wrapped linear weights and biases."""

    updated = model
    for path, wrapper in iter_ia3_modules(updated):
        if not wrapper.mergeable:
            raise ValueError("This IA3 module is not mergeable.")
        base = eqx.tree_at(
            lambda linear: linear.weight,
            wrapper.base,
            wrapper.base.weight * wrapper.ia3[:, None],
        )
        if base.bias is not None:
            base = eqx.tree_at(
                lambda linear: linear.bias,
                base,
                base.bias * wrapper.ia3,
            )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, base)
    return updated


def iter_ia3_modules(model: PyTree) -> tuple[tuple[Path, IA3Linear], ...]:
    """Return path/module pairs for IA3 wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, IA3Linear),
        )
        if isinstance(leaf, IA3Linear)
    )


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
    "iter_ia3_modules",
    "merge_ia3",
)
