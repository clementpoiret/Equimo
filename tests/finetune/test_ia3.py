"""IA3 tuning tests."""

from __future__ import annotations

import jax.numpy as jnp

import equimo.finetune as eqft


def test_ia3_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    tuned = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(target=eqft.TargetSpec(tags=("attention.proj",))),
    )

    assert jnp.allclose(tiny_vision_transformer(x), tuned(x), atol=1e-6)


def test_ia3_labels(tiny_vision_transformer):
    tuned = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(target=eqft.TargetSpec(tags=("attention.proj",))),
    )
    plan = eqft.prepare_finetune(
        tuned,
        trainable=eqft.TrainableSpec(mode="peft", method_name="ia3"),
    )

    assert plan.trainable.blocks[0].attn.proj.ia3 is not None
    assert plan.trainable.blocks[0].attn.proj.base.weight is None
    assert "ia3_decay" in plan.report.trainable_by_label
