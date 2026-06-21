"""Shared PEFT helpers."""

from __future__ import annotations

from typing import Protocol

from .._typing import PyTree


class PEFTModuleMixin(Protocol):
    """Protocol for PEFT wrappers that can merge adapter weights."""

    merged: bool

    def merge(self): ...

    def unmerge(self): ...


def get_path(tree: PyTree, path: tuple[str | int, ...]):
    """Resolve ``path`` in an Equinox PyTree/module."""

    node = tree
    for part in path:
        node = node[part] if isinstance(part, int) else getattr(node, part)
    return node


__all__ = (
    "PEFTModuleMixin",
    "get_path",
)
