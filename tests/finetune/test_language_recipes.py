"""Language recipe tests."""

from __future__ import annotations

import jax.random as jr

from equimo.finetune.language import recipes


def test_language_recipes_work_on_tiny_text(tiny_text_encoder):
    lora = recipes.lora_encoder(tiny_text_encoder, key=jr.PRNGKey(0), rank=2)
    prefix = recipes.prefix_encoder(tiny_text_encoder, key=jr.PRNGKey(1), num_prefix_tokens=2)
    head = recipes.projection_head(4, 3, key=jr.PRNGKey(2))
    frozen = recipes.locked_tower(tiny_text_encoder)

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert prefix.prefixes[0].shape == (2, 4)
    assert head.head.layers[-1].out_features == 3
    assert frozen.report.trainable_params == 0
