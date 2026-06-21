"""Scale/shift fine-tuning wrappers."""

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
class ScaleShiftConfig:
    """Configuration for feature-axis scale/shift tuning."""

    scale_init: float = 1.0
    shift_init: float = 0.0
    axis: Literal["feature", "channel"] = "feature"
    target: TargetSpec = TargetSpec(tags=("attention", "mlp", "norm"))
    train_head: bool = True
    mergeable: bool = True

    @classmethod
    def convnet(
        cls,
        *,
        target: TargetSpec = TargetSpec(tags=("conv", "stage.block", "norm")),
        axis: Literal["feature", "channel"] = "channel",
        scale_init: float = 1.0,
        shift_init: float = 0.0,
        train_head: bool = True,
        mergeable: bool = True,
    ) -> "ScaleShiftConfig":
        """Return the ConvNet scale/shift preset."""

        return cls(
            target=target,
            axis=axis,
            scale_init=scale_init,
            shift_init=shift_init,
            train_head=train_head,
            mergeable=mergeable,
        )


class ScaleShift(eqx.Module):
    """Trainable affine feature transform."""

    scale: jax.Array
    shift: jax.Array
    axis: Literal["feature", "channel"] = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        scale_init: float = 1.0,
        shift_init: float = 0.0,
        axis: Literal["feature", "channel"] = "feature",
        scale: jax.Array | None = None,
        shift: jax.Array | None = None,
    ):
        self.scale = jnp.full((dim,), scale_init, dtype=jnp.float32) if scale is None else scale
        self.shift = jnp.full((dim,), shift_init, dtype=jnp.float32) if shift is None else shift
        self.axis = axis

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.axis == "channel" and x.ndim > 1:
            shape = (self.scale.shape[0],) + (1,) * (x.ndim - 1)
            return x * self.scale.reshape(shape) + self.shift.reshape(shape)
        return x * self.scale + self.shift


class ScaleShiftWrapper(eqx.Module):
    """Wrap a module output with a trainable scale/shift transform."""

    base: eqx.Module
    scale_shift: ScaleShift
    mergeable: bool = eqx.field(static=True, default=True)

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
            mergeable=config.mergeable,
        )
        updated = eqx.tree_at(lambda tree, p=module_path: get_path(tree, p), updated, wrapper)

    return updated


def merge_scale_shift(model: PyTree) -> PyTree:
    """Fold scale/shift wrappers into linear modules where algebraically safe."""

    updated = model
    for path, wrapper in iter_scale_shift_wrappers(updated):
        if not wrapper.mergeable:
            raise ValueError("This scale/shift module is not mergeable.")
        base = wrapper.base
        if not isinstance(base, eqx.nn.Linear):
            raise ValueError(
                "Scale/shift merge is only algebraically safe for eqx.nn.Linear "
                f"wrappers; got {type(base).__name__} at {path_to_str(path)}."
            )
        scale = wrapper.scale_shift.scale.astype(base.weight.dtype)
        shift = wrapper.scale_shift.shift.astype(base.weight.dtype)
        merged = eqx.tree_at(
            lambda linear: linear.weight,
            base,
            base.weight * scale[:, None],
        )
        if merged.bias is not None:
            bias = merged.bias * scale + shift
            merged = eqx.tree_at(lambda linear: linear.bias, merged, bias)
        elif bool(jnp.any(shift != 0)):
            raise ValueError(
                "Cannot merge nonzero shift into a bias-free linear module at "
                f"{path_to_str(path)}."
            )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, merged)
    return updated


def iter_scale_shift_wrappers(
    model: PyTree,
) -> tuple[tuple[Path, ScaleShiftWrapper], ...]:
    """Return path/module pairs for scale/shift wrappers in ``model``."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, ScaleShiftWrapper),
        )
        if isinstance(leaf, ScaleShiftWrapper)
    )


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
    "iter_scale_shift_wrappers",
    "merge_scale_shift",
)
