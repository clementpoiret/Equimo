"""PiSSA initialization tests."""

from __future__ import annotations

import jax.random as jr

import equimo.finetune as eqft


def test_pissa_initialization_shapes(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.PiSSAConfig(
            rank=2,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    module = model.blocks[0].attn.proj

    assert module.lora_A.shape == (2, 4)
    assert module.lora_B.shape == (4, 2)
