"""Audio recipe tests."""

from __future__ import annotations

import jax.random as jr

from equimo.finetune.audio import recipes


def test_audio_recipes_work_on_tiny_ast(tiny_ast_like_encoder):
    lora = recipes.lora_ast(tiny_ast_like_encoder, key=jr.PRNGKey(0), rank=2)
    adapter = recipes.adapter_ast(tiny_ast_like_encoder, key=jr.PRNGKey(1), bottleneck=3)
    head = recipes.multilabel_tagging_head(4, 6, key=jr.PRNGKey(2))
    ctc = recipes.ctc_head(4, 8, key=jr.PRNGKey(3))

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert adapter.blocks[0].mlp.adapters[0].down.out_features == 3
    assert head.head.linear.out_features == 6
    assert ctc.head.linear.out_features == 8
