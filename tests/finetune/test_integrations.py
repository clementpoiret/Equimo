"""Optional integration helper tests."""

from __future__ import annotations

import equimo.finetune as eqft
from equimo.finetune.integrations import optax, rollfast


def test_dependency_free_integration_group_metadata(tiny_vision_transformer):
    plan = eqft.full_ft_llrd(tiny_vision_transformer)

    optax_groups = optax.optax_group_metadata(plan)
    rollfast_groups = rollfast.rollfast_group_metadata(plan)

    assert optax_groups == rollfast_groups
    assert any(group["label"] == "block_01_decay" for group in optax_groups)
    assert optax.optax_label_tree(plan) is plan.labels
    assert rollfast.rollfast_label_tree(plan) is plan.labels
