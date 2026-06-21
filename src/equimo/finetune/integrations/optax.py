"""Dependency-free Optax integration metadata helpers."""

from __future__ import annotations

from typing import Any

from ..config import FineTunePlan
from ._groups import group_metadata


def optax_group_metadata(plan: FineTunePlan) -> tuple[dict[str, Any], ...]:
    """Return group records suitable for external Optax transform builders."""

    return group_metadata(plan)


def optax_label_tree(plan: FineTunePlan) -> Any:
    """Return the parameter-label tree for external Optax masking/grouping."""

    return plan.labels


__all__ = (
    "optax_group_metadata",
    "optax_label_tree",
)
