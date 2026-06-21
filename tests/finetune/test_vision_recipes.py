"""Vision recipe tests."""

from __future__ import annotations

import jax.random as jr

from equimo.finetune.vision import recipes


def test_vision_recipes_work_on_tiny_vit(tiny_vision_transformer):
    lora = recipes.lora_vit(tiny_vision_transformer, key=jr.PRNGKey(0), rank=2)
    vpt = recipes.vpt_vit(tiny_vision_transformer, key=jr.PRNGKey(1), num_tokens=2)
    surgical = recipes.surgical_ft_vit(tiny_vision_transformer, shift="input")

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert vpt.prompts[0].shape == (2, 4)
    assert surgical.trainable.patch_embed.proj.weight is not None
