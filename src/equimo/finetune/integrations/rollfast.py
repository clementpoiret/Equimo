"""Dependency-free Rollfast integration metadata helpers."""

from __future__ import annotations

from typing import Any

from ..config import FineTunePlan
from ._groups import group_metadata


def rollfast_group_metadata(plan: FineTunePlan) -> tuple[dict[str, Any], ...]:
    """Return group records suitable for external Rollfast configuration."""

    return group_metadata(plan)


def rollfast_label_tree(plan: FineTunePlan) -> Any:
    """Return the parameter-label tree for external Rollfast grouping."""

    return plan.labels


__all__ = (
    "rollfast_group_metadata",
    "rollfast_label_tree",
)
