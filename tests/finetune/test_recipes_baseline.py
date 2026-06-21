"""Baseline fine-tuning recipe tests."""

from __future__ import annotations

import jax.random as jr

import equimo.finetune as eqft
from equimo.finetune.vision import recipes as vision_recipes

from fixtures import assert_tree_allclose


def test_lpft_stage_transition_preserves_head(tiny_vision_transformer):
    key = jr.PRNGKey(0)
    trained_like = eqft.replace_head(
        tiny_vision_transformer,
        eqft.LinearHead(4, 4, key=key),
    )
    recipe = eqft.lpft()

    stage2 = recipe.stage2_plan(trained_like)

    assert_tree_allclose(stage2.combine().head, trained_like.head)


def test_full_ft_llrd_recipe_freezes_patch_embed(tiny_vision_transformer):
    plan = eqft.full_ft_llrd(tiny_vision_transformer, decay=0.75)

    assert plan.trainable.patch_embed.proj.weight is None
    assert "block_01_decay" in plan.group_specs


def test_vision_partial_recipe_uses_last_blocks(tiny_vision_transformer):
    plan = vision_recipes.partial_ft_vit_llrd(
        tiny_vision_transformer,
        last_k_blocks=1,
    )

    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert plan.trainable.blocks[1].attn.qkv.weight is not None
