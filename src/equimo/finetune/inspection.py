"""Inspection helpers for fine-tuning plans."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import PyTree
from .config import FineTunePlan, ParamInfo, TrainableReport, TrainableSpec
from .paths import path_to_str


def inspect_trainables(
    model: PyTree | FineTunePlan,
    trainable: TrainableSpec | None = None,
) -> TrainableReport:
    """Return a trainability report for a model or existing plan."""

    if isinstance(model, FineTunePlan):
        return model.report

    from .surgery import prepare_finetune

    spec = TrainableSpec(mode="full") if trainable is None else trainable
    return prepare_finetune(model, trainable=spec).report


def inspect_plan(plan: FineTunePlan) -> TrainableReport:
    """Return the report stored on a prepared fine-tuning plan."""

    return plan.report


def make_trainable_report(model: PyTree, param_info: PyTree) -> TrainableReport:
    """Summarize trainable/frozen parameters from a ``ParamInfo`` tree."""

    leaves_by_path = {
        info.path: leaf
        for info, leaf in zip(
            _param_info_leaves(param_info),
            _param_leaves(model),
            strict=True,
        )
    }

    total_params = 0
    trainable_params = 0
    adapter_params = 0
    head_params = 0
    trainable_by_label: defaultdict[str, int] = defaultdict(int)
    frozen_by_label: defaultdict[str, int] = defaultdict(int)
    target_paths: list[str] = []
    estimated_delta_size_bytes = 0

    for info in _param_info_leaves(param_info):
        leaf = leaves_by_path[info.path]
        count = int(leaf.size)
        total_params += count

        if info.trainable:
            trainable_params += count
            trainable_by_label[info.label or "trainable"] += count
            target_paths.append(path_to_str(info.path))
            estimated_delta_size_bytes += count * int(leaf.dtype.itemsize)
            if "head" in info.tags:
                head_params += count
            if _is_adapter_leaf(info):
                adapter_params += count
        else:
            frozen_by_label[_frozen_label(info)] += count

    trainable_fraction = trainable_params / total_params if total_params else 0.0

    return TrainableReport(
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_fraction=trainable_fraction,
        trainable_by_label=dict(trainable_by_label),
        frozen_by_label=dict(frozen_by_label),
        adapter_params=adapter_params,
        head_params=head_params,
        mergeable=False,
        estimated_delta_size_bytes=estimated_delta_size_bytes,
        target_paths=tuple(target_paths),
    )


def _param_info_leaves(tree: Any) -> tuple[ParamInfo, ...]:
    return tuple(leaf for leaf in jtu.tree_leaves(tree) if isinstance(leaf, ParamInfo))


def _param_leaves(tree: PyTree) -> tuple[Any, ...]:
    return tuple(
        leaf
        for leaf in jtu.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
        if eqx.is_inexact_array(leaf)
    )


def _frozen_label(info: ParamInfo) -> str:
    if info.role:
        return info.role
    if info.depth is not None:
        return "block"
    return "frozen"


def _is_adapter_leaf(info: ParamInfo) -> bool:
    tags = info.tags
    return any(
        tag in tags
        for tag in (
            "adapter",
            "lora",
            "prompt",
            "prefix",
            "ia3",
            "scale_shift",
        )
    )


__all__ = (
    "inspect_plan",
    "inspect_trainables",
    "make_trainable_report",
)
