"""Path utilities for Equimo fine-tuning PyTrees."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import ParamInfo

LeafFilter = Callable[[Any], bool]


def key_path_to_path(key_path: tuple[Any, ...]) -> Path:
    """Convert a JAX key path into Equimo's stable path representation."""

    return tuple(_key_to_part(key) for key in key_path)


def path_to_str(path: Path) -> str:
    """Format a path as a dot-separated string."""

    return ".".join(str(part) for part in path)


def str_to_path(path: str) -> Path:
    """Parse a dot-separated path string."""

    if not path:
        return ()
    return tuple(_parse_path_part(part) for part in path.split("."))


def is_path_prefix(prefix: Path, path: Path) -> bool:
    """Return whether ``prefix`` identifies ``path`` or one of its parents."""

    return len(prefix) <= len(path) and path[: len(prefix)] == prefix


def iter_param_leaves(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> tuple[tuple[Path, Any], ...]:
    """Return stable paths and values for parameter-like leaves."""

    filtered = eqx.filter(tree, predicate)
    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(filtered)
        if predicate(leaf)
    )


def iter_param_paths(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> tuple[Path, ...]:
    """Return stable paths for parameter-like leaves."""

    return tuple(path for path, _ in iter_param_leaves(tree, predicate=predicate))


def extract_param_paths(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> tuple[str, ...]:
    """Return dot-formatted paths for parameter-like leaves."""

    return tuple(
        path_to_str(path) for path in iter_param_paths(tree, predicate=predicate)
    )


def make_path_tree(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> PyTree:
    """Replace parameter-like leaves with their stable paths."""

    filtered = eqx.filter(tree, predicate)
    return jtu.tree_map_with_path(
        lambda key_path, _: key_path_to_path(key_path),
        filtered,
    )


def make_param_info_tree(
    tree: PyTree,
    *,
    predicate: LeafFilter = eqx.is_inexact_array,
) -> PyTree:
    """Replace parameter-like leaves with base ``ParamInfo`` records."""

    filtered = eqx.filter(tree, predicate)
    return jtu.tree_map_with_path(_param_info_from_leaf, filtered)


def _param_info_from_leaf(key_path: tuple[Any, ...], leaf: Any) -> ParamInfo:
    return ParamInfo(
        path=key_path_to_path(key_path),
        is_array=eqx.is_array(leaf),
        is_inexact_array=eqx.is_inexact_array(leaf),
    )


def _key_to_part(key: Any) -> str | int:
    if isinstance(key, jtu.GetAttrKey):
        return key.name
    if isinstance(key, jtu.SequenceKey):
        return key.idx
    if isinstance(key, jtu.DictKey):
        return _normalise_part(key.key)

    flattened_index_key = getattr(jtu, "FlattenedIndexKey", None)
    if flattened_index_key is not None and isinstance(key, flattened_index_key):
        return _normalise_part(key.key)

    return _normalise_part(key)


def _normalise_part(part: Any) -> str | int:
    return part if isinstance(part, int) else str(part)


def _parse_path_part(part: str) -> str | int:
    return int(part) if part.isdecimal() else part


__all__ = (
    "LeafFilter",
    "extract_param_paths",
    "is_path_prefix",
    "iter_param_leaves",
    "iter_param_paths",
    "key_path_to_path",
    "make_param_info_tree",
    "make_path_tree",
    "path_to_str",
    "str_to_path",
)
