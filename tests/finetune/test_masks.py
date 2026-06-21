"""Fine-tuning trainability-mask tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

import equimo.finetune as eqft

from fixtures import assert_no_trainable_with_tag, assert_only_trainable_tags


def test_head_only_mask(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(mode="head"),
    )

    assert plan.trainable.head.weight is not None
    assert plan.trainable.head.bias is not None
    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert_only_trainable_tags(plan, {"head"})
    assert plan.report.trainable_params == 10


def test_head_plus_norm_mask(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(mode="head_plus_norm"),
    )

    assert plan.trainable.head.weight is not None
    assert plan.trainable.norm.weight is not None
    assert plan.trainable.blocks[0].norm1.weight is not None
    assert plan.trainable.blocks[0].mlp.fc1.weight is None
    assert_only_trainable_tags(plan, {"head", "norm"})


def test_bitfit_mask_trains_biases_plus_head(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(mode="bias", train_head=True),
    )

    assert plan.trainable.head.weight is not None
    assert plan.trainable.blocks[0].attn.qkv.bias is not None
    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert_only_trainable_tags(plan, {"bias", "head"})


def test_bitfit_mask_can_skip_head_weight(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(mode="bias", train_head=False),
    )

    assert plan.trainable.head.weight is None
    assert plan.trainable.head.bias is not None
    assert_only_trainable_tags(plan, {"bias"})


def test_bitfit_config_helper_builds_trainable_spec(tiny_vision_transformer):
    spec = eqft.bitfit_trainable_spec(
        eqft.BitFitConfig(train_head=False, include_norm_bias=False)
    )
    plan = eqft.prepare_finetune(tiny_vision_transformer, trainable=spec)

    assert plan.trainable.head.weight is None
    assert plan.trainable.head.bias is not None
    assert plan.trainable.blocks[0].norm1.bias is None
    assert plan.trainable.blocks[0].attn.qkv.bias is not None


def test_bitfit_config_can_skip_linear_biases(tiny_vision_transformer):
    spec = eqft.bitfit_trainable_spec(
        eqft.BitFitConfig(
            train_head=False,
            include_linear_bias=False,
            include_norm_bias=True,
        )
    )
    plan = eqft.prepare_finetune(tiny_vision_transformer, trainable=spec)

    assert plan.trainable.blocks[0].attn.qkv.bias is None
    assert plan.trainable.blocks[0].norm1.bias is not None
    assert plan.trainable.head.bias is None


def test_bitfit_config_can_include_positional_parameters(tiny_vision_transformer):
    spec = eqft.bitfit_trainable_spec(
        eqft.BitFitConfig(
            train_bias=False,
            train_head=False,
            include_positional_parameters=True,
        )
    )
    plan = eqft.prepare_finetune(tiny_vision_transformer, trainable=spec)

    assert plan.trainable.pos_embed is not None
    assert plan.trainable.blocks[0].attn.qkv.bias is None
    assert plan.trainable.head.bias is None


def test_full_ft_patch_embed_frozen(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="full",
            freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
        ),
    )

    assert plan.trainable.patch_embed.proj.weight is None
    assert plan.trainable.patch_embed.proj.bias is None
    assert plan.trainable.pos_embed is not None
    assert_no_trainable_with_tag(plan, "embedding.patch")


def test_frozen_patch_embed_does_not_change_after_one_step(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="full",
            freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
        ),
    )
    before = eqft.extract_subtree(
        tiny_vision_transformer,
        eqft.TargetSpec(tags_any=("embedding.patch",)),
    )

    def loss_fn(trainable):
        model = plan.combine(trainable)
        return jnp.sum(model(jnp.ones((2, 3))))

    _, grads = eqx.filter_value_and_grad(loss_fn)(plan.trainable)
    updates = jax.tree.map(
        lambda grad: -0.01 * grad if eqx.is_inexact_array(grad) else grad,
        grads,
    )
    after = plan.combine(eqx.apply_updates(plan.trainable, updates))

    assert before.patch_embed.proj.weight is not None
    assert after.patch_embed.proj.weight is not None
    assert jnp.array_equal(before.patch_embed.proj.weight, after.patch_embed.proj.weight)
    assert jnp.array_equal(before.patch_embed.proj.bias, after.patch_embed.proj.bias)


def test_partial_ft_last_block(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="partial",
            depth_range=(1, 2),
            train_head=True,
            train_norm=True,
        ),
    )

    assert plan.trainable.blocks[0].attn.qkv.weight is None
    assert plan.trainable.blocks[1].attn.qkv.weight is not None
    assert plan.trainable.head.weight is not None
    assert plan.trainable.norm.weight is not None


def test_surgical_input_feature_output_regions(tiny_vision_transformer):
    input_plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="surgical",
            method_name="input",
            train_head=False,
        ),
    )
    output_plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="surgical",
            method_name="output",
            train_head=True,
        ),
    )

    assert input_plan.trainable.patch_embed.proj.weight is not None
    assert input_plan.trainable.blocks[0].attn.qkv.weight is not None
    assert input_plan.trainable.blocks[1].attn.qkv.weight is None
    assert output_plan.trainable.blocks[0].attn.qkv.weight is None
    assert output_plan.trainable.blocks[1].attn.qkv.weight is not None
    assert output_plan.trainable.head.weight is not None
