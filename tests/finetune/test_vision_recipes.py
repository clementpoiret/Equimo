"""Vision recipe tests."""

from __future__ import annotations

import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu
import pytest

import equimo.finetune as eqft
from equimo.finetune.vision import dense
from equimo.finetune.vision import recipes
from equimo.vision.models.vit import dinov2_vitb14, dinov2_vits14


def _trainable_inner_block_indices(plan):
    return tuple(
        index
        for index, block in enumerate(plan.trainable.blocks[0].blocks)
        if any(eqx.is_inexact_array(leaf) for leaf in jtu.tree_leaves(block))
    )


def test_vision_recipes_work_on_tiny_vit(tiny_vision_transformer):
    lora = recipes.lora_vit(tiny_vision_transformer, key=jr.PRNGKey(0), rank=2)
    vpt = recipes.vpt_vit(tiny_vision_transformer, key=jr.PRNGKey(1), num_tokens=2)

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert vpt.prompts[0].shape == (2, 4)


@pytest.mark.parametrize("model_factory", (dinov2_vits14, dinov2_vitb14))
@pytest.mark.parametrize(
    ("last_k_blocks", "expected_indices"),
    (
        (1, (11,)),
        (3, (9, 10, 11)),
    ),
)
def test_dinov2_partial_ft_uses_final_inner_blocks(
    model_factory,
    last_k_blocks,
    expected_indices,
):
    model = model_factory(
        pretrained=False,
        img_size=28,
        dim=16,
        num_heads=[2],
        num_classes=0,
        key=jr.PRNGKey(0),
    )

    plan = eqft.partial_ft_last_k_blocks(
        model,
        k=last_k_blocks,
        train_head=False,
        train_norm=False,
        freeze_patch_embed=False,
    )

    assert len(model.blocks) == 1
    assert len(model.blocks[0].blocks) == 12
    assert _trainable_inner_block_indices(plan) == expected_indices


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
