"""Selector resolution for Equimo fine-tuning targets."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import ParamInfo, TargetSpec
from .paths import (
    is_path_prefix,
    iter_param_leaves,
    key_path_to_path,
    path_to_str,
)
from .tags import Tagger, canonical_tags_for_path, iter_param_infos

SelectorPredicate = Callable[[Path, Any], bool]


def resolve_target(
    model: PyTree,
    target: TargetSpec,
    *,
    allow_empty: bool = False,
    tagger: Tagger = canonical_tags_for_path,
) -> tuple[ParamInfo, ...]:
    """Resolve a ``TargetSpec`` to tagged parameter leaf metadata."""

    infos = iter_param_infos(model, tagger=tagger)
    by_path = {info.path: info for info in infos}
    selected_paths: set[Path] = set()

    has_positive_selector = bool(target.tags or target.include or target.predicate)
    if not has_positive_selector:
        selected_paths.update(by_path)

    if target.tags:
        selected_paths.update(
            info.path for info in infos if not info.tags.isdisjoint(target.tags)
        )

    if target.include:
        selected_paths.update(
            info.path
            for info in infos
            if _matches_any_pattern(info.path, target.include)
        )

    if target.predicate is not None:
        selected_paths.update(_resolve_predicate_paths(model, infos, target.predicate))

    selected_paths = {
        path
        for path in selected_paths
        if _depth_matches(by_path[path], target)
        and not _matches_any_pattern(path, target.exclude)
    }

    resolved = tuple(info for info in infos if info.path in selected_paths)
    if not resolved and not allow_empty:
        raise ValueError(_empty_selector_message(target))
    return resolved


def resolve_target_paths(
    model: PyTree,
    target: TargetSpec,
    *,
    allow_empty: bool = False,
    tagger: Tagger = canonical_tags_for_path,
) -> tuple[str, ...]:
    """Resolve a ``TargetSpec`` to stable dot-formatted parameter paths."""

    return tuple(
        path_to_str(info.path)
        for info in resolve_target(
            model,
            target,
            allow_empty=allow_empty,
            tagger=tagger,
        )
    )


def is_linear(path: Path, node: Any) -> bool:
    """Return whether ``node`` is an Equinox linear module."""

    del path
    return isinstance(node, eqx.nn.Linear)


def is_layer_norm(path: Path, node: Any) -> bool:
    """Return whether ``node`` is an Equinox LayerNorm module."""

    del path
    return isinstance(node, eqx.nn.LayerNorm)


def _resolve_predicate_paths(
    model: PyTree,
    infos: tuple[ParamInfo, ...],
    predicate: SelectorPredicate,
) -> set[Path]:
    leaves_by_path = dict(iter_param_leaves(model))
    selected: set[Path] = {
        info.path for info in infos if predicate(info.path, leaves_by_path[info.path])
    }

    for module_path, module in _iter_selector_modules(model):
        if not predicate(module_path, module):
            continue
        selected.update(
            info.path for info in infos if is_path_prefix(module_path, info.path)
        )

    return selected


def _matches_any_pattern(path: Path, patterns: Iterable[str]) -> bool:
    return any(_matches_pattern(path, pattern) for pattern in patterns)


def _matches_pattern(path: Path, pattern: str) -> bool:
    path_str = path_to_str(path)
    candidates = [pattern]
    if pattern.startswith("*."):
        candidates.append(pattern[2:])

    for candidate in candidates:
        if fnmatch.fnmatchcase(path_str, candidate):
            return True
        if fnmatch.fnmatchcase(path_str, f"{candidate}.*"):
            return True
    return False


def _depth_matches(info: ParamInfo, target: TargetSpec) -> bool:
    if target.min_depth is None and target.max_depth is None:
        return True
    if info.depth is None:
        return False
    if target.min_depth is not None and info.depth < target.min_depth:
        return False
    return target.max_depth is None or info.depth <= target.max_depth


def _empty_selector_message(target: TargetSpec) -> str:
    details: list[str] = []
    if target.tags:
        details.append(f"tags={target.tags!r}")
    if target.include:
        details.append(f"include={target.include!r}")
    if target.exclude:
        details.append(f"exclude={target.exclude!r}")
    if target.predicate is not None:
        details.append(f"predicate={_predicate_name(target.predicate)}")
    if target.min_depth is not None:
        details.append(f"min_depth={target.min_depth!r}")
    if target.max_depth is not None:
        details.append(f"max_depth={target.max_depth!r}")
    if not details:
        details.append("all parameter leaves")

    return (
        "TargetSpec resolved no parameter leaves for "
        f"{', '.join(details)}. Set allow_empty=True to permit empty targets."
    )


def _predicate_name(predicate: SelectorPredicate) -> str:
    return getattr(predicate, "__name__", predicate.__class__.__name__)


def _iter_selector_modules(model: PyTree) -> tuple[tuple[Path, Any], ...]:
    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=_is_selector_module,
        )
        if _is_selector_module(leaf)
    )


def _is_selector_module(node: Any) -> bool:
    return isinstance(node, _SELECTOR_MODULE_TYPES)


def _selector_module_types() -> tuple[type[Any], ...]:
    names = ("Linear", "LayerNorm", "Embedding", "Conv1d", "Conv2d", "Conv3d")
    return tuple(
        module_type
        for name in names
        if isinstance((module_type := getattr(eqx.nn, name, None)), type)
    )


_SELECTOR_MODULE_TYPES = _selector_module_types()


__all__ = (
    "SelectorPredicate",
    "is_layer_norm",
    "is_linear",
    "resolve_target",
    "resolve_target_paths",
)
