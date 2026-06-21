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


def test_head_plus_norm_recipe_accepts_config(tiny_vision_transformer):
    plan = eqft.head_plus_norm(
        tiny_vision_transformer,
        eqft.HeadPlusNormConfig(train_head=False, train_norm=True),
    )

    assert plan.trainable.head.weight is None
    assert plan.trainable.blocks[0].norm1.weight is not None


def test_head_plus_norm_config_controls_norm_scale_and_bias(tiny_vision_transformer):
    plan = eqft.head_plus_norm(
        tiny_vision_transformer,
        eqft.HeadPlusNormConfig(
            train_head=False,
            train_norm=True,
            train_norm_scale=False,
            train_norm_bias=True,
        ),
    )

    assert plan.trainable.blocks[0].norm1.weight is None
    assert plan.trainable.blocks[0].norm1.bias is not None


def test_head_plus_norm_config_can_include_embeddings(tiny_vision_transformer):
    plan = eqft.head_plus_norm(
        tiny_vision_transformer,
        eqft.HeadPlusNormConfig(
            train_head=False,
            train_norm=False,
            include_embeddings=True,
            include_positional_parameters=True,
        ),
    )

    assert plan.trainable.patch_embed.proj.weight is not None
    assert plan.trainable.pos_embed is not None
    assert plan.trainable.blocks[0].norm1.weight is None
    assert plan.trainable.head.weight is None


def test_partial_unfreeze_config_controls_span_and_embeddings(tiny_vision_transformer):
    plan = eqft.partial_unfreeze(
        tiny_vision_transformer,
        eqft.PartialUnfreezeConfig(
            fraction=0.5,
            train_embeddings=True,
            train_positional_parameters=True,
        ),
    )

    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert plan.trainable.blocks[1].attn.qkv.weight is not None
    assert plan.trainable.patch_embed.proj.weight is not None
    assert plan.trainable.pos_embed is not None


def test_surgical_config_controls_shift_span_and_embedding_policy(tiny_vision_transformer):
    plan = eqft.surgical(
        tiny_vision_transformer,
        eqft.HeuristicSurgicalPreset(
            shift="input",
            span_fraction=0.5,
            train_head=False,
            train_norm=False,
            train_embeddings_for_input_shift=False,
        ),
    )

    assert plan.trainable.patch_embed.proj.weight is None
    assert plan.trainable.blocks[0].attn.qkv.weight is not None
    assert plan.trainable.blocks[1].attn.qkv.weight is None
    assert plan.trainable.head.weight is None


def test_vision_partial_recipe_uses_last_blocks(tiny_vision_transformer):
    plan = vision_recipes.partial_ft_vit_llrd(
        tiny_vision_transformer,
        last_k_blocks=1,
    )

    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert plan.trainable.blocks[1].attn.qkv.weight is not None


def test_recommended_recipe_aliases_work(tiny_vision_transformer):
    lora = eqft.recipes.lora_transformer(
        tiny_vision_transformer,
        key=jr.PRNGKey(0),
        rank=2,
    )
    lora_all = eqft.recipes.lora_transformer_all_linear(
        tiny_vision_transformer,
        key=jr.PRNGKey(1),
        rank=2,
    )
    prompted = eqft.recipes.vpt_deep(
        tiny_vision_transformer,
        key=jr.PRNGKey(2),
        num_tokens=2,
    )
    bank = eqft.recipes.task_adapter_bank(
        tiny_vision_transformer,
        key=jr.PRNGKey(3),
        names=("task_a", "task_b"),
        bottleneck=3,
    )

    assert lora.blocks[0].attn.qkv.lora_A.shape[0] == 2
    assert lora_all.head.lora_A.shape[0] == 2
    assert prompted.prompts[0].shape == (2, 4)
    assert bank.blocks[0].mlp.adapter_names == ("task_a", "task_b")
