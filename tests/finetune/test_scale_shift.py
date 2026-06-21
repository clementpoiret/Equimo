"""Scale/shift tuning tests."""

from __future__ import annotations

import jax.numpy as jnp

import equimo.finetune as eqft


def test_scale_shift_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    tuned = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm", "*.norm1", "*.norm2"))),
    )

    assert jnp.allclose(tiny_vision_transformer(x), tuned(x), atol=1e-6)


def test_scale_shift_labels(tiny_vision_transformer):
    tuned = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm",))),
    )
    plan = eqft.prepare_finetune(
        tuned,
        trainable=eqft.TrainableSpec(mode="peft", method_name="scale_shift"),
    )

    assert plan.trainable.norm.scale_shift.scale is not None
    assert plan.trainable.norm.base.weight is None
    assert "scale_shift_no_decay" in plan.report.trainable_by_label
