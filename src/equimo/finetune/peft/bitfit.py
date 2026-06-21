"""BitFit mask configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._typing import Path
from ..config import TargetSpec, TrainableSpec
from ..tags import canonical_tags_for_path


@dataclass(frozen=True)
class BitFitConfig:
    """Configuration for BitFit trainability masks."""

    train_bias: bool = True
    train_head: bool = True
    include_norm_bias: bool = True
    include_linear_bias: bool = True
    include_conv_bias: bool = True
    include_positional_parameters: bool = False


def bitfit_trainable_spec(config: BitFitConfig | None = None) -> TrainableSpec:
    """Return a ``TrainableSpec`` for BitFit-style fine-tuning."""

    config = BitFitConfig() if config is None else config
    if config.train_bias or config.include_positional_parameters:
        return TrainableSpec(
            mode="surgical",
            target=TargetSpec(
                predicate=lambda path, leaf: _bitfit_predicate(path, leaf, config)
            ),
            train_head=config.train_head,
        )
    if config.train_head:
        return TrainableSpec(mode="head")
    return TrainableSpec(mode="frozen")


def _bitfit_predicate(path: Path, leaf: Any, config: BitFitConfig) -> bool:
    tags = canonical_tags_for_path(path, leaf)
    if config.include_positional_parameters and "embedding.position" in tags:
        return True
    if not config.train_bias or "bias" not in tags:
        return False
    if "norm" in tags:
        return config.include_norm_bias
    if _is_conv_path(path):
        return config.include_conv_bias
    return config.include_linear_bias


def _is_conv_path(path: Path) -> bool:
    parts = {str(part).lower() for part in path}
    return any("conv" in part for part in parts)


__all__ = (
    "BitFitConfig",
    "bitfit_trainable_spec",
)
