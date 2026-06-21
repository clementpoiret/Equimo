"""IA3 tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

import equimo.finetune as eqft


def test_ia3_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    tuned = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(target=eqft.TargetSpec(tags_any=("attention.proj",))),
    )

    assert jnp.allclose(tiny_vision_transformer(x), tuned(x), atol=1e-6)


def test_ia3_default_targets_fused_key_value_and_mlp_hidden(tiny_vision_transformer):
    tuned = eqft.apply_ia3(tiny_vision_transformer)
    qkv = tuned.blocks[0].attn.qkv
    width = qkv.base.out_features // 3

    assert isinstance(qkv, eqft.IA3Linear)
    assert tuple(segment.name for segment in qkv.projection_segments) == ("k", "v")
    assert qkv.ia3.shape == (2 * width,)
    assert jnp.array_equal(qkv.scale_vector()[:width], jnp.ones((width,)))
    assert isinstance(tuned.blocks[0].mlp.fc1, eqft.IA3Linear)
    assert not isinstance(tuned.blocks[0].attn.proj, eqft.IA3Linear)


def test_merge_ia3_preserves_outputs_and_removes_wrapper(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    tuned = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(target=eqft.TargetSpec(tags_any=("attention.proj",))),
    )
    tuned = eqx.tree_at(
        lambda model: model.blocks[0].attn.proj.ia3,
        tuned,
        jnp.asarray([0.5, 1.0, 1.5, 2.0]),
    )

    merged = eqft.merge_ia3(tuned)

    assert not isinstance(merged.blocks[0].attn.proj, eqft.IA3Linear)
    assert jnp.allclose(tuned(x), merged(x), atol=1e-6)


def test_merge_ia3_fused_qkv_preserves_query_rows(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    tuned = eqft.apply_ia3(tiny_vision_transformer)
    qkv = tuned.blocks[0].attn.qkv
    width = qkv.base.out_features // 3
    ia3 = jnp.concatenate(
        [
            jnp.full((width,), 2.0),
            jnp.full((width,), 3.0),
        ]
    )
    tuned = eqx.tree_at(lambda model: model.blocks[0].attn.qkv.ia3, tuned, ia3)

    merged = eqft.merge_ia3(tuned)
    original_weight = tiny_vision_transformer.blocks[0].attn.qkv.weight
    merged_weight = merged.blocks[0].attn.qkv.weight

    assert not isinstance(merged.blocks[0].attn.qkv, eqft.IA3Linear)
    assert jnp.array_equal(merged_weight[:width], original_weight[:width])
    assert jnp.allclose(
        merged_weight[width : 2 * width], original_weight[width : 2 * width] * 2.0
    )
    assert jnp.allclose(merged_weight[2 * width :], original_weight[2 * width :] * 3.0)
    assert jnp.allclose(tuned(x), merged(x), atol=1e-6)


def test_merge_ia3_rejects_non_mergeable_wrappers(tiny_vision_transformer):
    tuned = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            mergeable=False,
        ),
    )

    with pytest.raises(ValueError, match="not mergeable"):
        eqft.merge_ia3(tuned)


def test_ia3_labels(tiny_vision_transformer):
    tuned = eqft.apply_ia3(
        tiny_vision_transformer,
        eqft.IA3Config(target=eqft.TargetSpec(tags_any=("attention.proj",))),
    )
    plan = eqft.prepare_finetune(
        tuned,
        trainable=eqft.TrainableSpec(mode="peft", method_name="ia3"),
    )

    assert plan.trainable.blocks[0].attn.proj.ia3 is not None
    assert plan.trainable.blocks[0].attn.proj.base.weight is None
    assert "ia3_decay" in plan.report.trainable_by_label
