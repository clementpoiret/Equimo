"""Fine-tuning inspection/report tests."""

from __future__ import annotations

import equimo.finetune as eqft


def test_report_counts(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(mode="head"),
    )

    report = eqft.inspect_plan(plan)
    assert report.total_params == 394
    assert report.trainable_params == 10
    assert report.head_params == 10
    assert report.trainable_fraction == 10 / 394
    assert report.trainable_by_label == {
        "head_decay": 8,
        "head_no_decay": 2,
    }
    assert report.target_paths == ("head.weight", "head.bias")


def test_inspect_trainables_prepares_default_full_plan(tiny_vision_transformer):
    report = eqft.inspect_trainables(tiny_vision_transformer)

    assert report.total_params == 394
    assert report.trainable_params == 394
