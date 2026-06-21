"""Parameter labels and learning-rate metadata for fine-tuning plans."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import GroupSpec, LLRDConfig, ParamInfo
from .paths import key_path_to_path
from .tags import Tagger, canonical_tags_for_path, make_param_info


def make_param_labels(
    model: PyTree,
    llrd_config: LLRDConfig | None = None,
    *,
    trainable_paths: frozenset[Path] | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Return a PyTree of optimizer-group labels for parameter leaves."""

    info_tree, _ = make_labeled_param_info_tree(
        model,
        llrd_config,
        trainable_paths=trainable_paths,
        tagger=tagger,
    )
    return jtu.tree_map(_label_from_info, info_tree)


def make_labeled_param_info_tree(
    model: PyTree,
    llrd_config: LLRDConfig | None = None,
    *,
    trainable_paths: frozenset[Path] | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> tuple[PyTree, dict[str, GroupSpec]]:
    """Build a ``ParamInfo`` tree with trainability, labels, and group specs."""

    config = LLRDConfig(decay=1.0) if llrd_config is None else llrd_config
    all_depths = _all_depths(model, tagger=tagger)
    selected_depths = _selected_depths(model, trainable_paths, tagger=tagger)
    filtered = eqx.filter(model, eqx.is_inexact_array)
    group_specs: dict[str, GroupSpec] = {}

    def make_info(key_path: tuple[Any, ...], leaf: Any) -> ParamInfo | None:
        if not eqx.is_inexact_array(leaf):
            return None

        path = key_path_to_path(key_path)
        base = make_param_info(path, leaf, tagger=tagger)
        trainable = trainable_paths is None or path in trainable_paths
        weight_decay = trainable and _uses_weight_decay(base, config)
        lr_multiplier = (
            _lr_multiplier(base, config, all_depths, selected_depths)
            if trainable
            else None
        )
        label = (
            _label_for_info(base, config, weight_decay)
            if trainable
            else None
        )

        info = replace(
            base,
            trainable=trainable,
            weight_decay=weight_decay,
            lr_multiplier=lr_multiplier,
            label=label,
        )
        if trainable and label is not None and lr_multiplier is not None:
            group_specs.setdefault(
                label,
                GroupSpec(
                    label=label,
                    role=info.role,
                    depth=info.depth,
                    lr_multiplier=lr_multiplier,
                    weight_decay=weight_decay,
                    tags=tuple(sorted(info.tags)),
                ),
            )
        return info

    return jtu.tree_map_with_path(make_info, filtered), group_specs


def _label_from_info(info: ParamInfo | None) -> str | None:
    return info.label if isinstance(info, ParamInfo) else None


def _uses_weight_decay(info: ParamInfo, config: LLRDConfig) -> bool:
    return info.tags.isdisjoint(config.no_weight_decay_tags)


def _label_for_info(
    info: ParamInfo,
    config: LLRDConfig,
    weight_decay: bool,
) -> str:
    suffix = "decay" if weight_decay else "no_decay"
    return f"{_label_base(info, config)}_{suffix}"


def _label_base(info: ParamInfo, config: LLRDConfig) -> str:
    if "lora.A" in info.tags:
        return "lora_A"
    if "lora.B" in info.tags:
        return "lora_B"
    if "lora" in info.tags:
        return "lora"
    if "dora.magnitude" in info.tags:
        return "dora_magnitude"
    if "dora" in info.tags:
        return "dora"
    if "adapter" in info.tags:
        return "adapter"
    if "prompt" in info.tags:
        return "prompt"
    if "prefix" in info.tags:
        return "prefix"
    if "scale_shift" in info.tags:
        return "scale_shift"
    if "ia3" in info.tags:
        return "ia3"
    if "vera" in info.tags:
        return "vera"
    if "side_tuning" in info.tags:
        return "side_tuning"
    if "head" in info.tags:
        return config.head_label
    if info.depth is not None:
        return config.block_label_format.format(depth=info.depth)
    if any(tag.startswith("embedding.") for tag in info.tags):
        return config.embedding_label
    if "norm" in info.tags:
        return "norm"
    if "bias" in info.tags:
        return "bias"
    return "param"


def _lr_multiplier(
    info: ParamInfo,
    config: LLRDConfig,
    all_depths: tuple[int, ...],
    selected_depths: tuple[int, ...],
) -> float:
    if "head" in info.tags:
        return float(config.head_lr_multiplier)

    max_depth = None
    if config.rebase_selected_depth and selected_depths:
        max_depth = max(selected_depths)
    elif all_depths:
        max_depth = max(all_depths)

    if info.depth is not None and max_depth is not None:
        return float(config.top_block_lr_multiplier * config.decay ** (max_depth - info.depth))

    if any(tag.startswith("embedding.") for tag in info.tags):
        if config.embedding_lr_multiplier is not None:
            return float(config.embedding_lr_multiplier)
        if max_depth is not None:
            return float(config.top_block_lr_multiplier * config.decay ** (max_depth + 1))

    return 1.0


def _all_depths(model: PyTree, *, tagger: Tagger) -> tuple[int, ...]:
    filtered = eqx.filter(model, eqx.is_inexact_array)
    depths: set[int] = set()
    for key_path, leaf in jtu.tree_leaves_with_path(filtered):
        if not eqx.is_inexact_array(leaf):
            continue
        depth = make_param_info(key_path_to_path(key_path), leaf, tagger=tagger).depth
        if depth is not None:
            depths.add(depth)
    return tuple(sorted(depths))


def _selected_depths(
    model: PyTree,
    trainable_paths: frozenset[Path] | None,
    *,
    tagger: Tagger,
) -> tuple[int, ...]:
    if trainable_paths is None:
        return _all_depths(model, tagger=tagger)

    filtered = eqx.filter(model, eqx.is_inexact_array)
    depths: set[int] = set()
    for key_path, leaf in jtu.tree_leaves_with_path(filtered):
        if not eqx.is_inexact_array(leaf):
            continue
        path = key_path_to_path(key_path)
        if path not in trainable_paths:
            continue
        depth = make_param_info(path, leaf, tagger=tagger).depth
        if depth is not None:
            depths.add(depth)
    return tuple(sorted(depths))


__all__ = (
    "make_labeled_param_info_tree",
    "make_param_labels",
)
