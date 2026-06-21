"""Tabular recipe tests."""

from __future__ import annotations

from equimo.finetune.tabular import recipes


def test_tabular_recipes_work_on_tiny_mlp(tiny_linear_mlp):
    frozen = recipes.freeze_all(tiny_linear_mlp)
    report = recipes.inspect_trainables(tiny_linear_mlp)

    assert frozen.report.trainable_params == 0
    assert report.total_params == 43
