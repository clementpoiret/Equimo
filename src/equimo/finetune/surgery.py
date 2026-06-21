"""Fine-tuning PyTree partitioning and model-side planning."""

from __future__ import annotations

from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp

from ._typing import PyTree
from .config import FineTunePlan, LLRDConfig, TargetSpec, TrainableSpec
from .inspection import make_trainable_report
from .labels import make_labeled_param_info_tree
from .masks import make_trainable_filter, resolve_trainable_paths
from .selectors import resolve_target
from .tags import Tagger, canonical_tags_for_path


class _HeadWithMetadata(eqx.Module):
    head: eqx.Module
    old_head_metadata: Mapping[str, Any] = eqx.field(static=True)

    def __call__(self, *args, **kwargs):
        return self.head(*args, **kwargs)


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
    sample_features: jax.Array | None = None,
    validate_shape: bool = True,
    preserve_old_head_metadata: bool = False,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Return ``model`` with one selected head module replaced."""

    if isinstance(selector, str):
        if hasattr(model, selector):
            path = (selector,)
        else:
            selector = TargetSpec(tags=(selector,))
            path = _resolve_single_module_path(model, selector, tagger=tagger)
    else:
        path = _resolve_single_module_path(model, selector, tagger=tagger)

    if not path:
        raise ValueError("replace_head cannot replace the model root.")
    old_head = _get_path(model, path)
    if validate_shape:
        _validate_head_replacement(old_head, head, sample_features)
    replacement = (
        _HeadWithMetadata(head, _head_metadata(old_head))
        if preserve_old_head_metadata
        else head
    )
    return eqx.tree_at(lambda m: _get_path(m, path), model, replacement)


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


def _resolve_single_module_path(
    model: PyTree,
    selector: TargetSpec,
    *,
    tagger: Tagger,
) -> tuple[str | int, ...]:
    paths = {
        info.path[:-1]
        for info in resolve_target(model, selector, allow_empty=False, tagger=tagger)
    }
    if len(paths) != 1:
        raise ValueError(
            "replace_head requires exactly one target module; "
            f"resolved {len(paths)} module paths."
        )
    return next(iter(paths))


def _validate_head_replacement(
    old_head: Any,
    new_head: Any,
    sample_features: jax.Array | None,
) -> None:
    old_in = _head_in_features(old_head)
    new_in = _head_in_features(new_head)
    if old_in is not None and new_in is not None and old_in != new_in:
        raise ValueError(
            "replace_head input-feature mismatch: "
            f"old head expects {old_in}, new head expects {new_in}."
        )

    if sample_features is None:
        return
    if sample_features.ndim == 0:
        raise ValueError("replace_head sample_features must have a feature axis.")
    feature_dim = int(sample_features.shape[-1])
    expected = new_in if new_in is not None else old_in
    if expected is not None and feature_dim != expected:
        raise ValueError(
            "replace_head sample feature dimension mismatch: "
            f"sample has {feature_dim}, selected head expects {expected}."
        )
    _validate_head_call(new_head, sample_features)


def _validate_head_call(head: Any, sample_features: jax.Array) -> None:
    try:
        output = head(sample_features)
    except Exception as error:
        raise ValueError(
            "replace_head could not call the replacement head with sample_features."
        ) from error
    if not eqx.is_array(output):
        raise ValueError(
            "replace_head sample validation expected the replacement head to return an array."
        )
    if jnp.asarray(output).shape == ():
        raise ValueError(
            "replace_head sample validation expected a non-scalar head output."
        )


def _head_in_features(head: Any) -> int | None:
    if isinstance(head, _HeadWithMetadata):
        return _head_in_features(head.head)
    if isinstance(head, eqx.nn.Linear):
        return int(head.in_features)
    linear = getattr(head, "linear", None)
    if isinstance(linear, eqx.nn.Linear):
        return int(linear.in_features)
    projection = getattr(head, "projection", None)
    if isinstance(projection, eqx.nn.Linear):
        return int(projection.in_features)
    nested_head = getattr(head, "head", None)
    if nested_head is not None and nested_head is not head:
        nested = _head_in_features(nested_head)
        if nested is not None:
            return nested
    layers = getattr(head, "layers", None)
    if layers:
        first = layers[0]
        if isinstance(first, eqx.nn.Linear):
            return int(first.in_features)
    return None


def _head_out_features(head: Any) -> int | None:
    if isinstance(head, _HeadWithMetadata):
        return _head_out_features(head.head)
    if isinstance(head, eqx.nn.Linear):
        return int(head.out_features)
    linear = getattr(head, "linear", None)
    if isinstance(linear, eqx.nn.Linear):
        return int(linear.out_features)
    projection = getattr(head, "projection", None)
    if isinstance(projection, eqx.nn.Linear):
        return int(projection.out_features)
    nested_head = getattr(head, "head", None)
    if nested_head is not None and nested_head is not head:
        nested = _head_out_features(nested_head)
        if nested is not None:
            return nested
    layers = getattr(head, "layers", None)
    if layers:
        last = layers[-1]
        if isinstance(last, eqx.nn.Linear):
            return int(last.out_features)
    return None


def _head_metadata(head: Any) -> dict[str, Any]:
    return {
        "class_name": head.__class__.__name__,
        "module": head.__class__.__module__,
        "in_features": _head_in_features(head),
        "out_features": _head_out_features(head),
    }


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
