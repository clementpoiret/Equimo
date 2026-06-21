"""LoRA+ label tests."""

from __future__ import annotations

import jax.random as jr

import equimo.finetune as eqft


def test_lora_plus_labels_expose_a_and_b(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="peft", method_name="lora", train_head=False),
    )

    assert "lora_A_decay" in plan.group_specs
    assert "lora_B_decay" in plan.group_specs
