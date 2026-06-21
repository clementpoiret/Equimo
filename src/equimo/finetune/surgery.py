"""Fine-tuning PyTree partitioning and model-side planning."""

from __future__ import annotations

import equinox as eqx

from ._typing import PyTree
from .config import FineTunePlan, LLRDConfig, TargetSpec, TrainableSpec
from .inspection import make_trainable_report
from .labels import make_labeled_param_info_tree
from .masks import make_trainable_filter, resolve_trainable_paths
from .selectors import resolve_target
from .tags import Tagger, canonical_tags_for_path


def prepare_finetune(
    model: PyTree,
    *,
    trainable: TrainableSpec,
    labels: LLRDConfig | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> FineTunePlan:
    """Partition a model and attach labels/reports for external optimizers."""

    trainable_paths = resolve_trainable_paths(model, trainable, tagger=tagger)
    trainable_mask = make_trainable_filter(model, trainable, tagger=tagger)
    trainable_tree, frozen_tree = eqx.partition(model, trainable_mask)
    param_info, group_specs = make_labeled_param_info_tree(
        model,
        labels,
        trainable_paths=trainable_paths,
        tagger=tagger,
    )
    label_tree = _labels_from_param_info(param_info)
    report = make_trainable_report(model, param_info)

    return FineTunePlan(
        trainable=trainable_tree,
        frozen=frozen_tree,
        labels=label_tree,
        group_specs=group_specs,
        trainable_mask=trainable_mask,
        param_info=param_info,
        report=report,
    )


def partition_for_training(
    model: PyTree,
    trainable: TrainableSpec,
    *,
    labels: LLRDConfig | None = None,
    tagger: Tagger = canonical_tags_for_path,
) -> FineTunePlan:
    """Alias for ``prepare_finetune``."""

    return prepare_finetune(
        model,
        trainable=trainable,
        labels=labels,
        tagger=tagger,
    )


def replace_head(
    model: PyTree,
    head: eqx.Module,
    *,
    selector: str | TargetSpec = "head",
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Return ``model`` with one selected head module replaced."""

    if isinstance(selector, str):
        if hasattr(model, selector):
            return eqx.tree_at(lambda m: getattr(m, selector), model, head)
        selector = TargetSpec(tags=(selector,))

    paths = {
        info.path[:-1]
        for info in resolve_target(model, selector, allow_empty=False, tagger=tagger)
    }
    if len(paths) != 1:
        raise ValueError(
            "replace_head requires exactly one target module; "
            f"resolved {len(paths)} module paths."
        )

    path = next(iter(paths))
    if not path:
        raise ValueError("replace_head cannot replace the model root.")
    return eqx.tree_at(lambda m: _get_path(m, path), model, head)


def extract_subtree(
    model: PyTree,
    target: TargetSpec,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Return a tree containing only leaves selected by ``target``."""

    plan = prepare_finetune(
        model,
        trainable=TrainableSpec(mode="peft", target=target),
        tagger=tagger,
    )
    return plan.trainable


def _get_path(tree: PyTree, path: tuple[str | int, ...]):
    node = tree
    for part in path:
        node = node[part] if isinstance(part, int) else getattr(node, part)
    return node


def _labels_from_param_info(param_info: PyTree) -> PyTree:
    import jax.tree_util as jtu

    from .config import ParamInfo

    return jtu.tree_map(
        lambda info: info.label if isinstance(info, ParamInfo) else None,
        param_info,
    )


__all__ = (
    "extract_subtree",
    "partition_for_training",
    "prepare_finetune",
    "replace_head",
)
