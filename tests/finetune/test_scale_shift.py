"""Scale/shift tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

import equimo.finetune as eqft


def test_scale_shift_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    tuned = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm", "*.norm1", "*.norm2"))),
    )

    assert jnp.allclose(tiny_vision_transformer(x), tuned(x), atol=1e-6)


def test_scale_shift_default_targets_attention_mlp_and_norm(tiny_vision_transformer):
    tuned = eqft.apply_scale_shift(tiny_vision_transformer)

    assert isinstance(tuned.blocks[0].attn.qkv, eqft.ScaleShiftWrapper)
    assert isinstance(tuned.blocks[0].attn.proj, eqft.ScaleShiftWrapper)
    assert isinstance(tuned.blocks[0].mlp.fc1, eqft.ScaleShiftWrapper)
    assert isinstance(tuned.blocks[0].mlp.fc2, eqft.ScaleShiftWrapper)
    assert isinstance(tuned.blocks[0].norm1, eqft.ScaleShiftWrapper)


def test_merge_scale_shift_preserves_linear_outputs_and_removes_wrapper(
    tiny_vision_transformer,
):
    x = jnp.ones((2, 3))
    tuned = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(tags_any=("attention.proj",))),
    )
    tuned = eqx.tree_at(
        lambda model: (
            model.blocks[0].attn.proj.scale_shift.scale,
            model.blocks[0].attn.proj.scale_shift.shift,
        ),
        tuned,
        (
            jnp.asarray([0.5, 1.0, 1.5, 2.0]),
            jnp.asarray([0.1, -0.2, 0.3, -0.4]),
        ),
    )

    merged = eqft.merge_scale_shift(tuned)

    assert not isinstance(merged.blocks[0].attn.proj, eqft.ScaleShiftWrapper)
    assert jnp.allclose(tuned(x), merged(x), atol=1e-6)


def test_merge_scale_shift_rejects_non_linear_wrappers(tiny_vision_transformer):
    tuned = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm",))),
    )

    with pytest.raises(ValueError, match="only algebraically safe"):
        eqft.merge_scale_shift(tuned)


def test_merge_scale_shift_rejects_non_mergeable_wrappers(tiny_vision_transformer):
    tuned = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            mergeable=False,
        ),
    )

    with pytest.raises(ValueError, match="not mergeable"):
        eqft.merge_scale_shift(tuned)


def test_scale_shift_channel_axis_and_convnet_preset():
    config = eqft.ScaleShiftConfig.convnet()
    transform = eqft.ScaleShift(
        3,
        axis="channel",
        scale=jnp.asarray([1.0, 2.0, 3.0]),
        shift=jnp.asarray([0.0, 1.0, 2.0]),
    )
    x = jnp.ones((3, 2, 2))

    y = transform(x)

    assert config.axis == "channel"
    assert config.target.tags_any == ("conv", "stage.block", "norm")
    assert jnp.array_equal(y[:, 0, 0], jnp.asarray([1.0, 3.0, 5.0]))


def test_scale_shift_labels(tiny_vision_transformer):
    tuned = eqft.apply_scale_shift(
        tiny_vision_transformer,
        eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm",))),
    )
    plan = eqft.prepare_finetune(
        tuned,
        trainable=eqft.TrainableSpec(mode="peft", method_name="scale_shift"),
    )

    assert plan.trainable.norm.scale_shift.scale is not None
    assert plan.trainable.norm.base.weight is None
    assert "scale_shift_no_decay" in plan.report.trainable_by_label
