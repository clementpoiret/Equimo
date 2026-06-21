"""Vision recipe tests."""

from __future__ import annotations

import jax.random as jr

from equimo.finetune.vision import dense
from equimo.finetune.vision import recipes


def test_vision_recipes_work_on_tiny_vit(tiny_vision_transformer):
    lora = recipes.lora_vit(tiny_vision_transformer, key=jr.PRNGKey(0), rank=2)
    vpt = recipes.vpt_vit(tiny_vision_transformer, key=jr.PRNGKey(1), num_tokens=2)

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert vpt.prompts[0].shape == (2, 4)


def test_vision_dense_utilities_create_adapter_and_distillation_config(finetune_key):
    adapter = dense.dense_feature_adapter(
        4,
        2,
        key=finetune_key,
        config=dense.DenseVisionConfig(activation="relu"),
    )
    config = dense.dense_distillation_config(normalize_features=False)

    assert adapter.projection.in_features == 4
    assert adapter.projection.out_features == 2
    assert adapter.activation == "relu"
    assert config.layers == ("25%", "50%", "75%", "100%")
    assert config.normalize_features is False
