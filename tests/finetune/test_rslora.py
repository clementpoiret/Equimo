"""rsLoRA tests."""

from __future__ import annotations

import pytest
import jax.random as jr

import equimo.finetune as eqft


def test_rslora_scaling(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.RsLoRAConfig(
            rank=4,
            alpha=16.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    assert model.blocks[0].attn.proj.scaling == pytest.approx(8.0)
