"""Fine-tuning parameter-label tests."""

from __future__ import annotations

import jax.tree_util as jtu

import equimo.finetune as eqft


def _infos(plan):
    return [
        leaf
        for leaf in jtu.tree_leaves(plan.param_info)
        if isinstance(leaf, eqft.ParamInfo)
    ]


def test_weight_decay_labels(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(mode="full"),
        labels=eqft.LLRDConfig(decay=0.75),
    )

    for info in _infos(plan):
        if "norm" in info.tags or "bias" in info.tags:
            assert info.weight_decay is False
            assert info.label is not None
            assert info.label.endswith("_no_decay")


def test_labels_match_trainable_tree(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="full",
            freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
        ),
        labels=eqft.LLRDConfig(decay=0.75),
    )

    for info in _infos(plan):
        if info.trainable:
            assert info.label is not None
        else:
            assert info.label is None


def test_make_param_labels_direct(tiny_vision_transformer):
    labels = eqft.make_param_labels(
        tiny_vision_transformer,
        eqft.LLRDConfig(decay=0.75),
    )

    assert labels.head.weight == "head_decay"
    assert labels.head.bias == "head_no_decay"
    assert labels.blocks[1].attn.qkv.weight == "block_01_decay"
