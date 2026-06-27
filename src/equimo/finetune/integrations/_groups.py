"""Shared optimizer-group metadata helpers."""

from __future__ import annotations

from typing import Any

from ..config import FineTunePlan


def group_metadata(plan: FineTunePlan) -> tuple[dict[str, Any], ...]:
    """Return optimizer-group metadata records without creating an optimizer."""

    return tuple(
        {
            "label": label,
            "role": group.role,
            "depth": group.depth,
            "lr_multiplier": group.lr_multiplier,
            "weight_decay": group.weight_decay,
            "tags": group.tags,
            "tags_all": group.tags_all,
            "roles": group.roles,
            "mixed_roles": group.mixed_roles,
        }
        for label, group in sorted(plan.group_specs.items())
    )


__all__ = ("group_metadata",)
