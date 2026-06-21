"""Audio selector adapters for fine-tuning tags."""

from __future__ import annotations

from typing import Any

from .._typing import Path, PyTree
from ..config import ParamInfo, TargetSpec
from ..selectors import resolve_target as _resolve_target
from ..selectors import resolve_target_paths as _resolve_target_paths
from ..tags import canonical_tags_for_path, make_tag_tree as _make_tag_tree


def audio_tags_for_path(path: Path, leaf: Any | None = None) -> frozenset[str]:
    """Infer semantic tags for audio encoder models."""

    parts = tuple(str(part) for part in path)
    tags = set(canonical_tags_for_path(path, leaf))

    if "patch_embed" in parts:
        tags.add("audio.patch_embed")
    if "blocks" in parts:
        tags.add("audio.transformer.block")
    if "head" in parts:
        tags.add("audio.tagging_head")
    if "norm" in parts:
        tags.add("audio.pooling")

    return frozenset(tags)


def make_tag_tree(model: PyTree) -> PyTree:
    """Build an audio-tagged ``ParamInfo`` tree."""

    return _make_tag_tree(model, tagger=audio_tags_for_path)


def resolve_target(
    model: PyTree,
    target: TargetSpec,
    *,
    allow_empty: bool = False,
) -> tuple[ParamInfo, ...]:
    """Resolve a target using audio semantic tags."""

    return _resolve_target(
        model,
        target,
        allow_empty=allow_empty,
        tagger=audio_tags_for_path,
    )


def resolve_target_paths(
    model: PyTree,
    target: TargetSpec,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Resolve a target to dot paths using audio semantic tags."""

    return _resolve_target_paths(
        model,
        target,
        allow_empty=allow_empty,
        tagger=audio_tags_for_path,
    )


__all__ = (
    "audio_tags_for_path",
    "make_tag_tree",
    "resolve_target",
    "resolve_target_paths",
)
