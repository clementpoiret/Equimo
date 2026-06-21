"""Scale/shift fine-tuning wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
class ScaleShiftConfig:
    """Configuration for feature-axis scale/shift tuning."""

    scale_init: float = 1.0
    shift_init: float = 0.0
    axis: Literal["feature"] = "feature"
    target: TargetSpec = TargetSpec(tags=("norm",))


class ScaleShift(eqx.Module):
    """Trainable affine feature transform."""

    scale: jax.Array
    shift: jax.Array
    axis: Literal["feature"] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        scale_init: float = 1.0,
        shift_init: float = 0.0,
        axis: Literal["feature"] = "feature",
        scale: jax.Array | None = None,
        shift: jax.Array | None = None,
    ):
        self.scale = jnp.full((dim,), scale_init, dtype=jnp.float32) if scale is None else scale
        self.shift = jnp.full((dim,), shift_init, dtype=jnp.float32) if shift is None else shift
        self.axis = axis

    def __call__(self, x: jax.Array) -> jax.Array:
        return x * self.scale + self.shift


class ScaleShiftWrapper(eqx.Module):
    """Wrap a module output with a trainable scale/shift transform."""

    base: eqx.Module
    scale_shift: ScaleShift

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        y = _call_base(self.base, x, *args, key=key, inference=inference, **kwargs)
        return self.scale_shift(y)


def apply_scale_shift(
    model: PyTree,
    config: ScaleShiftConfig | None = None,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Apply scale/shift wrappers to selected modules."""

    config = ScaleShiftConfig() if config is None else config
    module_paths = _target_module_paths(model, config.target, tagger=tagger)
    updated = model

    for module_path in module_paths:
        module = get_path(updated, module_path)
        if isinstance(module, ScaleShiftWrapper):
            continue
        dim = _infer_output_dim(module)
        wrapper = ScaleShiftWrapper(
            module,
            ScaleShift(
                dim,
                scale_init=config.scale_init,
                shift_init=config.shift_init,
                axis=config.axis,
            ),
        )
        updated = eqx.tree_at(lambda tree, p=module_path: get_path(tree, p), updated, wrapper)

    return updated


def _target_module_paths(model: PyTree, target: TargetSpec, *, tagger: Tagger) -> tuple[Path, ...]:
    paths = {
        info.path[:-1]
        for info in resolve_target(model, target, tagger=tagger)
        if info.path[-1:] in (("weight",), ("bias",), ("scale",))
    }
    return tuple(sorted(paths, key=path_to_str))


def _infer_output_dim(module: eqx.Module) -> int:
    if isinstance(module, eqx.nn.Linear):
        return int(module.out_features)
    if hasattr(module, "shape"):
        shape = module.shape
        return int(shape[0] if isinstance(shape, tuple) else shape)
    if hasattr(module, "weight"):
        return int(module.weight.shape[0])
    raise ValueError(f"Could not infer scale/shift dimension for {type(module).__name__}.")


def _call_base(base, x, *args, key, inference, **kwargs):
    call_kwargs = dict(kwargs)
    if key is not None:
        call_kwargs["key"] = key
    if inference is not None:
        call_kwargs["inference"] = inference
    try:
        return base(x, *args, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        call_kwargs.pop("key", None)
        return base(x, *args, **call_kwargs)


__all__ = (
    "ScaleShift",
    "ScaleShiftConfig",
    "ScaleShiftWrapper",
    "apply_scale_shift",
)
