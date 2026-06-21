"""Semantic tags for fine-tuning selector infrastructure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import ParamInfo
from .paths import iter_param_leaves, key_path_to_path

Tagger = Callable[[Path, Any], Iterable[str]]

CANONICAL_TAGS: tuple[str, ...] = (
    "embedding.patch",
    "embedding.position",
    "embedding.class_token",
    "block",
    "attention.qkv",
    "attention.proj",
    "mlp.fc1",
    "mlp.fc2",
    "norm",
    "head",
)


def canonical_tags_for_path(path: Path, leaf: Any | None = None) -> frozenset[str]:
    """Infer canonical semantic tags from an Equimo-style parameter path."""

    del leaf
    parts = tuple(str(part) for part in path)
    tags: set[str] = set()

    if "patch_embed" in parts:
        tags.add("embedding.patch")
    if any(
        part in {"pos_embed", "position_embed", "position_embedding"} for part in parts
    ):
        tags.add("embedding.position")
    if "cls_token" in parts:
        tags.add("embedding.class_token")
    if "dist_token" in parts:
        tags.add("embedding.distillation_token")
    if "mask_token" in parts:
        tags.add("embedding.mask_token")

    depth = infer_depth(path)
    if depth is not None:
        tags.add("block")
        tags.add(f"block.{depth}")

    _add_attention_tags(parts, tags)
    _add_mlp_tags(parts, tags)
    _add_norm_tags(parts, tags)

    if "head" in parts or "classifier" in parts:
        tags.add("head")

    if parts[-1:] == ("bias",):
        tags.add("bias")
    elif parts[-1:] in (("weight",), ("scale",)):
        tags.add("weight")

    return frozenset(tags)


def infer_depth(path: Path) -> int | None:
    """Infer semantic block depth from common Equimo path shapes."""

    parts = tuple(str(part) for part in path)
    for index, part in enumerate(parts[:-1]):
        if part in {"blocks", "block"}:
            depth = _maybe_int(parts[index + 1])
            if depth is not None:
                return depth

    for index, part in enumerate(parts[:-1]):
        if part == "stages":
            stage = _maybe_int(parts[index + 1])
            if stage is not None:
                return stage

    return None


def make_tag_tree(
    tree: PyTree,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Replace parameter-like leaves with tagged ``ParamInfo`` records."""

    filtered = eqx.filter(tree, eqx.is_inexact_array)

    def make_info(key_path: tuple[Any, ...], leaf: Any) -> ParamInfo:
        return make_param_info(key_path_to_path(key_path), leaf, tagger=tagger)

    return jtu.tree_map_with_path(make_info, filtered)


def iter_param_infos(
    tree: PyTree,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> tuple[ParamInfo, ...]:
    """Return tagged ``ParamInfo`` records for inexact array leaves."""

    return tuple(
        make_param_info(path, leaf, tagger=tagger)
        for path, leaf in iter_param_leaves(tree)
    )


def make_param_info(
    path: Path,
    leaf: Any,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> ParamInfo:
    """Build one tagged ``ParamInfo`` record."""

    tags = frozenset(tagger(path, leaf))
    return ParamInfo(
        path=path,
        tags=tags,
        role=infer_role(tags),
        depth=infer_depth(path),
        is_array=eqx.is_array(leaf),
        is_inexact_array=eqx.is_inexact_array(leaf),
    )


def infer_role(tags: Iterable[str]) -> str:
    """Return a coarse role name from semantic tags."""

    tag_set = frozenset(tags)
    for tag, role in (
        ("head", "head"),
        ("attention.qkv", "attention.qkv"),
        ("attention.proj", "attention.proj"),
        ("mlp.fc1", "mlp.fc1"),
        ("mlp.fc2", "mlp.fc2"),
        ("norm", "norm"),
        ("embedding.patch", "embedding.patch"),
        ("embedding.position", "embedding.position"),
        ("embedding.class_token", "embedding.class_token"),
        ("block", "block"),
    ):
        if tag in tag_set:
            return role
    return ""


def merge_taggers(*taggers: Tagger) -> Tagger:
    """Combine multiple taggers into one tagger."""

    def merged(path: Path, leaf: Any) -> frozenset[str]:
        tags: set[str] = set()
        for tagger in taggers:
            tags.update(tagger(path, leaf))
        return frozenset(tags)

    return merged


def _add_attention_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    for index, part in enumerate(parts[:-1]):
        if part not in {"attn", "attention", "self_attn", "self_attention"}:
            continue
        next_part = parts[index + 1]
        if next_part == "qkv":
            tags.add("attention.qkv")
        elif next_part in {"proj", "out_proj", "projection"}:
            tags.add("attention.proj")
        elif next_part in {"q", "query"}:
            tags.add("attention.q")
        elif next_part in {"k", "key"}:
            tags.add("attention.k")
        elif next_part in {"v", "value"}:
            tags.add("attention.v")


def _add_mlp_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    mlp_like = bool({"mlp", "ffn", "feed_forward"}.intersection(parts))
    if not mlp_like:
        return
    if "fc1" in parts:
        tags.add("mlp.fc1")
    if "fc2" in parts:
        tags.add("mlp.fc2")


def _add_norm_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    norm_parts = {"norm", "norm1", "norm2", "ln", "layer_norm"}
    matched = norm_parts.intersection(parts)
    if not matched:
        return

    tags.add("norm")
    if "norm1" in matched:
        tags.add("block.norm.pre")
    if "norm2" in matched:
        tags.add("block.norm.post")


def _maybe_int(value: str) -> int | None:
    return int(value) if value.isdecimal() else None


__all__ = (
    "CANONICAL_TAGS",
    "Tagger",
    "canonical_tags_for_path",
    "infer_depth",
    "infer_role",
    "iter_param_infos",
    "make_param_info",
    "make_tag_tree",
    "merge_taggers",
)
