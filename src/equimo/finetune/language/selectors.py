"""Language selector adapters for fine-tuning tags."""

from __future__ import annotations

from typing import Any

from .._typing import Path, PyTree
from ..config import ParamInfo, TargetSpec
from ..selectors import resolve_target as _resolve_target
from ..selectors import resolve_target_paths as _resolve_target_paths
from ..tags import canonical_tags_for_path, make_tag_tree as _make_tag_tree


def language_tags_for_path(path: Path, leaf: Any | None = None) -> frozenset[str]:
    """Infer semantic tags for text encoder models."""

    parts = tuple(str(part) for part in path)
    tags = set(canonical_tags_for_path(path, leaf))

    if "token_embed" in parts:
        tags.update(("text.embedding", "text.token"))
    if "pos_embed" in parts:
        tags.add("text.position")
    if "blocks" in parts:
        tags.add("text.block")
    if "attention.qkv" in tags:
        tags.add("text.attention.qkv")
    if "attention.proj" in tags:
        tags.add("text.attention.proj")
    if "mlp.fc1" in tags:
        tags.add("text.mlp.fc1")
    if "mlp.fc2" in tags:
        tags.add("text.mlp.fc2")
    if "head" in tags:
        tags.add("text.projection_head")

    return frozenset(tags)


def make_tag_tree(model: PyTree) -> PyTree:
    """Build a language-tagged ``ParamInfo`` tree."""

    return _make_tag_tree(model, tagger=language_tags_for_path)


def resolve_target(
    model: PyTree,
    target: TargetSpec,
    *,
    allow_empty: bool = False,
) -> tuple[ParamInfo, ...]:
    """Resolve a target using language semantic tags."""

    return _resolve_target(
        model,
        target,
        allow_empty=allow_empty,
        tagger=language_tags_for_path,
    )


def resolve_target_paths(
    model: PyTree,
    target: TargetSpec,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Resolve a target to dot paths using language semantic tags."""

    return _resolve_target_paths(
        model,
        target,
        allow_empty=allow_empty,
        tagger=language_tags_for_path,
    )


__all__ = (
    "language_tags_for_path",
    "make_tag_tree",
    "resolve_target",
    "resolve_target_paths",
)
