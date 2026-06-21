"""Tabular fine-tuning recipe presets."""

from __future__ import annotations

from .._typing import PyTree
from ..config import FineTunePlan, TrainableReport, TrainableSpec
from ..inspection import inspect_trainables as _inspect_trainables
from ..surgery import prepare_finetune


def freeze_all(model: PyTree) -> FineTunePlan:
    """Freeze all tabular model leaves."""

    return prepare_finetune(model, trainable=TrainableSpec(mode="frozen"))


def head_only(model: PyTree) -> FineTunePlan:
    """Train only tabular head leaves."""

    return prepare_finetune(model, trainable=TrainableSpec(mode="head"))


def inspect_trainables(model: PyTree) -> TrainableReport:
    """Inspect tabular trainability."""

    return _inspect_trainables(model)


__all__ = (
    "freeze_all",
    "head_only",
    "inspect_trainables",
)
