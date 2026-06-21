"""Fine-tuning trainability-mask tests."""

from __future__ import annotations

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


def test_full_ft_patch_embed_frozen(tiny_vision_transformer):
    plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="full",
            freeze=eqft.TargetSpec(tags=("embedding.patch",)),
        ),
    )

    assert plan.trainable.patch_embed.proj.weight is None
    assert plan.trainable.patch_embed.proj.bias is None
    assert plan.trainable.pos_embed is not None
    assert_no_trainable_with_tag(plan, "embedding.patch")


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
            shift="input",
            train_head=False,
        ),
    )
    output_plan = eqft.prepare_finetune(
        tiny_vision_transformer,
        trainable=eqft.TrainableSpec(
            mode="surgical",
            shift="output",
            train_head=True,
        ),
    )

    assert input_plan.trainable.patch_embed.proj.weight is not None
    assert input_plan.trainable.blocks[0].attn.qkv.weight is not None
    assert input_plan.trainable.blocks[1].attn.qkv.weight is None
    assert output_plan.trainable.blocks[0].attn.qkv.weight is None
    assert output_plan.trainable.blocks[1].attn.qkv.weight is not None
    assert output_plan.trainable.head.weight is not None
