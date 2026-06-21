"""Layer-wise learning-rate decay tests."""

from __future__ import annotations

import pytest
import jax.random as jr

import equimo.finetune as eqft

from fixtures import TinyConvNeXtLike, TinyVisionTransformer


def test_llrd_multipliers_12_blocks():
    model = TinyVisionTransformer(depth=12, key=jr.PRNGKey(0))
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="full",
            freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
        ),
        labels=eqft.LLRDConfig(decay=0.75),
    )

    assert plan.group_specs["block_11_decay"].lr_multiplier == pytest.approx(1.0)
    assert plan.group_specs["block_10_decay"].lr_multiplier == pytest.approx(0.75)
    assert plan.group_specs["block_09_decay"].lr_multiplier == pytest.approx(0.75**2)
    assert plan.group_specs["block_00_decay"].lr_multiplier == pytest.approx(0.75**11)
    assert "block_11_no_decay" in plan.group_specs
    assert plan.trainable.patch_embed.proj.weight is None


def test_partial_llrd_keeps_original_depth_indices():
    model = TinyVisionTransformer(depth=12, key=jr.PRNGKey(0))
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="partial",
            depth_range=(8, 12),
            train_head=True,
            freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
        ),
        labels=eqft.LLRDConfig(decay=0.75, rebase_selected_depth=False),
    )

    assert "block_08_decay" in plan.group_specs
    assert "block_07_decay" not in plan.group_specs
    assert plan.group_specs["block_08_decay"].lr_multiplier == pytest.approx(0.75**3)
    assert plan.group_specs["block_11_decay"].lr_multiplier == pytest.approx(1.0)


def test_llrd_depth_axis_stage_uses_stage_indices():
    model = TinyConvNeXtLike(stage_depths=(1, 1, 1), key=jr.PRNGKey(0))
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
        labels=eqft.LLRDConfig(decay=0.5, depth_axis="stage"),
    )

    assert plan.group_specs["block_02_decay"].lr_multiplier == pytest.approx(1.0)
    assert plan.group_specs["block_01_decay"].lr_multiplier == pytest.approx(0.5)
    assert plan.group_specs["block_00_decay"].lr_multiplier == pytest.approx(0.25)


def test_llrd_rejects_invalid_depth_axis(tiny_vision_transformer):
    with pytest.raises(ValueError, match="depth_axis"):
        eqft.prepare_finetune(
            tiny_vision_transformer,
            trainable=eqft.TrainableSpec(mode="full"),
            labels=eqft.LLRDConfig(depth_axis="layer"),  # type: ignore[arg-type]
        )
