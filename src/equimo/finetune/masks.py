"""Trainability masks for Equimo fine-tuning plans."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import ParamInfo, TargetSpec, TrainableSpec
from .paths import key_path_to_path
from .selectors import resolve_target
from .tags import Tagger, canonical_tags_for_path, iter_param_infos


def make_trainable_filter(
    model: PyTree,
    spec: TrainableSpec,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Build an Equinox filter PyTree for the requested trainability policy."""

    trainable_paths = resolve_trainable_paths(model, spec, tagger=tagger)
    def is_trainable(key_path: tuple[Any, ...], leaf: Any) -> bool:
        if not eqx.is_inexact_array(leaf):
            return False
        return key_path_to_path(key_path) in trainable_paths

    return jtu.tree_map_with_path(is_trainable, model)


def resolve_trainable_paths(
    model: PyTree,
    spec: TrainableSpec,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> frozenset[Path]:
    """Resolve a ``TrainableSpec`` to stable parameter paths."""

    infos = iter_param_infos(model, tagger=tagger)
    selected = _base_trainable_paths(model, infos, spec, tagger=tagger)
    selected.update(_flag_trainable_paths(infos, spec))
    if spec.mode == "peft":
        selected.update(_trainable_peft_base_paths(model))

    if spec.freeze is not None:
        selected.difference_update(_target_paths(model, spec.freeze, tagger=tagger))

    selected.difference_update(_paths_with_any_tag(infos, {"peft.metadata"}))
    selected.difference_update(_nontrainable_peft_scale_paths(model))

    return frozenset(selected)


def _base_trainable_paths(
    model: PyTree,
    infos: tuple[ParamInfo, ...],
    spec: TrainableSpec,
    *,
    tagger: Tagger,
) -> set[Path]:
    mode = spec.mode

    if mode == "frozen":
        return set()
    if mode == "head":
        return _paths_with_any_tag(infos, {"head"})
    if mode == "head_plus_norm":
        return _paths_with_any_tag(infos, {"head", "norm"})
    if mode == "norm":
        return _paths_with_any_tag(infos, {"norm"})
    if mode == "bias":
        return _paths_with_any_tag(infos, {"bias"})
    if mode == "scale_shift":
        return _paths_with_any_tag(infos, {"scale_shift", "peft.scale_shift"})
    if mode == "peft":
        if spec.target is not None:
            return _target_paths(model, spec.target, tagger=tagger)
        tags = {
            "peft",
            "lora",
            "dora",
            "adapter",
            "randlora",
            "prompt",
            "prefix",
            "ia3",
            "scale_shift",
            "vera",
        }
        if spec.method_name is not None:
            tags.add(spec.method_name)
            tags.add(f"peft.{spec.method_name}")
        return _paths_with_any_tag(infos, tags)
    if mode == "partial":
        selected = _paths_in_depth_range(infos, spec.depth_range)
        if spec.target is not None:
            selected.intersection_update(_target_paths(model, spec.target, tagger=tagger))
        return selected
    if mode == "surgical":
        if spec.target is not None:
            return _target_paths(model, spec.target, tagger=tagger)
        return _surgical_paths(infos, spec.method_name)
    if mode == "full":
        selected = {info.path for info in infos}
        if not spec.train_head:
            selected.difference_update(_paths_with_any_tag(infos, {"head"}))
        return selected

    raise ValueError(f"Unsupported fine-tuning mode: {mode!r}")


def _flag_trainable_paths(
    infos: tuple[ParamInfo, ...],
    spec: TrainableSpec,
) -> set[Path]:
    if spec.mode == "frozen":
        return set()

    selected: set[Path] = set()

    if spec.mode not in {"head", "head_plus_norm", "full"} and spec.train_head:
        selected.update(_paths_with_any_tag(infos, {"head"}))
    if spec.mode not in {"head_plus_norm", "norm", "full"} and spec.train_norm:
        selected.update(_paths_with_any_tag(infos, {"norm"}))
    if spec.mode not in {"bias", "full"} and spec.train_bias:
        selected.update(_paths_with_any_tag(infos, {"bias"}))

    return selected


def _target_paths(model: PyTree, target: TargetSpec, *, tagger: Tagger) -> set[Path]:
    return {
        info.path
        for info in resolve_target(model, target, allow_empty=False, tagger=tagger)
    }


def _paths_with_any_tag(
    infos: tuple[ParamInfo, ...],
    tags: set[str],
) -> set[Path]:
    return {info.path for info in infos if not info.tags.isdisjoint(tags)}


def _paths_in_depth_range(
    infos: tuple[ParamInfo, ...],
    depth_range: tuple[int, int] | None,
) -> set[Path]:
    depths = [info.depth for info in infos if info.depth is not None]
    if not depths:
        return set()

    if depth_range is None:
        start, stop = max(depths), max(depths) + 1
    else:
        start, stop = depth_range

    return {
        info.path
        for info in infos
        if info.depth is not None and start <= info.depth < stop
    }


def _surgical_paths(
    infos: tuple[ParamInfo, ...],
    shift: str | None,
) -> set[Path]:
    depths = sorted({info.depth for info in infos if info.depth is not None})
    if not depths:
        return set()

    first_depth = depths[0]
    middle_depth = depths[len(depths) // 2]
    last_depth = depths[-1]
    shift = "output" if shift is None else shift

    if shift == "input":
        return {
            info.path
            for info in infos
            if info.depth == first_depth or "embedding.patch" in info.tags
        }
    if shift == "corruption":
        return {
            info.path
            for info in infos
            if info.depth == first_depth and ("norm" in info.tags or "block" in info.tags)
        }
    if shift in {"feature", "subpopulation"}:
        return {info.path for info in infos if info.depth == middle_depth}
    if shift in {"output", "label_space"}:
        return {info.path for info in infos if info.depth == last_depth}

    raise ValueError(
        "Unsupported surgical shift "
        f"{shift!r}; expected input, feature, output, corruption, "
        "subpopulation, or label_space."
    )


def _trainable_peft_base_paths(model: PyTree) -> set[Path]:
    paths: set[Path] = set()
    for key_path, wrapper in jtu.tree_leaves_with_path(
        model,
        is_leaf=_is_trainable_base_wrapper,
    ):
        if not _is_trainable_base_wrapper(wrapper):
            continue
        wrapper_path = key_path_to_path(key_path)
        base = wrapper.base
        filtered = eqx.filter(base, eqx.is_inexact_array)
        for base_key_path, leaf in jtu.tree_leaves_with_path(filtered):
            if eqx.is_inexact_array(leaf):
                paths.add((*wrapper_path, "base", *key_path_to_path(base_key_path)))
    return paths


def _nontrainable_peft_scale_paths(model: PyTree) -> set[Path]:
    paths: set[Path] = set()
    for key_path, node in jtu.tree_leaves_with_path(
        model,
        is_leaf=_has_nontrainable_peft_scale,
    ):
        if _has_nontrainable_peft_scale(node):
            paths.add((*key_path_to_path(key_path), "scale"))
    return paths


def _has_nontrainable_peft_scale(node: Any) -> bool:
    return (
        hasattr(node, "scale")
        and hasattr(node, "scale_trainable")
        and not bool(getattr(node, "scale_trainable"))
    )


def _is_trainable_base_wrapper(node: Any) -> bool:
    return bool(getattr(node, "train_base", False)) and hasattr(node, "base")


__all__ = (
    "make_trainable_filter",
    "resolve_trainable_paths",
)
