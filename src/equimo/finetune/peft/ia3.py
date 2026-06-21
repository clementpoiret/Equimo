"""IA3 activation scaling wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import ProjectionSegment, TargetSpec
from ..paths import key_path_to_path, path_to_str
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path


@dataclass(frozen=True)
class IA3Config:
    """Configuration for IA3 activation scaling."""

    target: TargetSpec = TargetSpec(
        tags_any=("attention.k", "attention.v", "mlp.hidden")
    )
    init: float = 1.0
    axis: Literal["feature"] = "feature"
    train_head: bool = True
    mergeable: bool = True


class IA3Linear(eqx.Module):
    """Wrap a linear module with trainable output-channel IA3 scaling."""

    base: eqx.nn.Linear
    ia3: jax.Array
    projection_segments: tuple[ProjectionSegment, ...] = eqx.field(static=True)
    mergeable: bool = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.nn.Linear,
        *,
        init: float = 1.0,
        ia3: jax.Array | None = None,
        projection_segments: tuple[ProjectionSegment, ...] = (),
        mergeable: bool = True,
    ):
        self.base = base
        self.projection_segments = tuple(projection_segments)
        self.mergeable = mergeable
        expected_dim = _ia3_dim(base, self.projection_segments)
        self.ia3 = (
            jnp.full((expected_dim,), init, dtype=base.weight.dtype)
            if ia3 is None
            else ia3
        )
        if self.ia3.shape != (expected_dim,):
            raise ValueError(
                "IA3 scale vector has shape "
                f"{self.ia3.shape}, expected ({expected_dim},)."
            )

    def scale_vector(self) -> jax.Array:
        """Return a full output-channel scale vector for this wrapped linear."""

        if not self.projection_segments:
            return self.ia3
        scale = jnp.ones((self.base.out_features,), dtype=self.ia3.dtype)
        offset = 0
        for segment in self.projection_segments:
            width = segment.stop - segment.start
            scale = scale.at[segment.start : segment.stop].set(
                self.ia3[offset : offset + width]
            )
            offset += width
        return scale

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.base(x) * self.scale_vector()


def _ia3_dim(
    base: eqx.nn.Linear,
    projection_segments: tuple[ProjectionSegment, ...],
) -> int:
    if not projection_segments:
        return int(base.out_features)
    dim = 0
    for segment in projection_segments:
        if segment.axis != 0:
            raise ValueError(
                "IA3 projection segments currently use logical output axis 0."
            )
        if (
            segment.start < 0
            or segment.stop > base.out_features
            or segment.start >= segment.stop
        ):
            raise ValueError(
                "IA3 projection segment "
                f"{segment.name!r} has invalid bounds {segment.start}:{segment.stop} "
                f"for output dimension {base.out_features}."
            )
        dim += segment.stop - segment.start
    return dim


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
    module_specs = _target_linear_specs(model, config.target, tagger=tagger)
    updated = model
    for module_path, projection_segments in module_specs:
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
            IA3Linear(
                module,
                init=config.init,
                projection_segments=projection_segments,
                mergeable=config.mergeable,
            ),
        )
    return updated


def merge_ia3(model: PyTree) -> PyTree:
    """Fold IA3 output scales into wrapped linear weights and biases."""

    updated = model
    for path, wrapper in iter_ia3_modules(updated):
        if not wrapper.mergeable:
            raise ValueError("This IA3 module is not mergeable.")
        scale = wrapper.scale_vector().astype(wrapper.base.weight.dtype)
        base = eqx.tree_at(
            lambda linear: linear.weight,
            wrapper.base,
            wrapper.base.weight * scale[:, None],
        )
        if base.bias is not None:
            base = eqx.tree_at(
                lambda linear: linear.bias,
                base,
                base.bias * scale,
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


def _target_linear_specs(
    model: PyTree,
    target: TargetSpec,
    *,
    tagger: Tagger,
) -> tuple[tuple[Path, tuple[ProjectionSegment, ...]], ...]:
    specs: dict[Path, tuple[ProjectionSegment, ...]] = {}
    resolved = resolve_target(
        model,
        target,
        allow_empty=_target_mentions_qkv_segment(target),
        tagger=tagger,
    )
    for info in resolved:
        if info.path[-1:] in (("weight",), ("bias",)):
            specs[info.path[:-1]] = ()

    if _target_mentions_qkv_segment(target):
        fused = resolve_target(
            model,
            TargetSpec(
                tags_any=("attention.qkv", "block.attention.qkv"),
                allow_empty=True,
            ),
            allow_empty=True,
            tagger=tagger,
        )
        for info in fused:
            if info.path[-1:] not in (("weight",), ("bias",)):
                continue
            module_path = info.path[:-1]
            if module_path in specs and not specs[module_path]:
                continue
            module = get_path(model, module_path)
            if not isinstance(module, eqx.nn.Linear):
                continue
            segments = _projection_segments_for_target(module, target)
            if segments:
                specs[module_path] = _merge_segments(
                    specs.get(module_path, ()),
                    segments,
                )

    if not specs and not target.allow_empty:
        raise ValueError("TargetSpec resolved no IA3 linear modules.")
    return tuple(sorted(specs.items(), key=lambda item: path_to_str(item[0])))


def _target_mentions_qkv_segment(target: TargetSpec) -> bool:
    tags = set(target.tags_all) | set(target.tags_any)
    suffixes = (".q", ".k", ".v")
    return any(
        tag in {"attention.q", "attention.k", "attention.v"} or tag.endswith(suffixes)
        for tag in tags
    )


def _projection_segments_for_target(
    module: eqx.nn.Linear,
    target: TargetSpec,
) -> tuple[ProjectionSegment, ...]:
    selected = _selected_qkv_segment_names(target)
    if not selected:
        return ()
    if module.out_features % 3 != 0:
        raise ValueError(
            "QKV projection segments require an output dimension divisible by 3."
        )
    width = module.out_features // 3
    starts = {"q": 0, "k": width, "v": 2 * width}
    return tuple(
        ProjectionSegment(
            name=name, axis=0, start=starts[name], stop=starts[name] + width
        )
        for name in ("q", "k", "v")
        if name in selected
    )


def _selected_qkv_segment_names(target: TargetSpec) -> frozenset[str]:
    names: set[str] = set()
    for tag in (*target.tags_all, *target.tags_any):
        last = tag.rsplit(".", maxsplit=1)[-1]
        if last in {"q", "k", "v"}:
            names.add(last)
    return frozenset(names)


def _merge_segments(
    existing: tuple[ProjectionSegment, ...],
    new: tuple[ProjectionSegment, ...],
) -> tuple[ProjectionSegment, ...]:
    by_name = {segment.name: segment for segment in existing}
    by_name.update({segment.name: segment for segment in new})
    return tuple(by_name[name] for name in ("q", "k", "v") if name in by_name)


__all__ = (
    "IA3Config",
    "IA3Linear",
    "apply_ia3",
    "iter_ia3_modules",
    "merge_ia3",
)
