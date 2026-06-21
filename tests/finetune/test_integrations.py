"""Optional integration helper tests."""

from __future__ import annotations

import equimo.finetune as eqft
import jax.tree_util as jtu
from equimo.finetune.integrations import optax, rollfast


def test_dependency_free_integration_group_metadata(tiny_vision_transformer):
    plan = eqft.full_ft_llrd(tiny_vision_transformer)

    optax_groups = optax.optax_group_metadata(plan)
    rollfast_groups = rollfast.rollfast_group_metadata(plan)

    assert optax_groups == rollfast_groups
    assert any(group["label"] == "block_01_decay" for group in optax_groups)
    assert optax.optax_label_tree(plan) is plan.labels
    assert rollfast.rollfast_label_tree(plan) is plan.labels


def test_rollfast_metadata_exposes_structural_compiler_fields(tiny_vision_transformer):
    plan = eqft.full_ft_llrd(tiny_vision_transformer)
    groups = rollfast.rollfast_group_metadata(plan)
    labels = {
        label
        for label in jtu.tree_leaves(plan.labels)
        if label is not None
    }
    group_labels = {group["label"] for group in groups}

    assert labels <= group_labels
    for group in groups:
        assert set(group) == {
            "label",
            "role",
            "depth",
            "lr_multiplier",
            "weight_decay",
            "tags",
        }
        assert group["lr_multiplier"] > 0.0
