"""Feature extraction and head replacement tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft

from fixtures import assert_tree_allclose


class NoKwargFeatureModel(eqx.Module):
    def features(self, x):
        return jnp.stack([x, x + 1.0])


class ConvFeatureModel(eqx.Module):
    stem: tuple[()] = eqx.field(static=True)
    stages: tuple[()] = eqx.field(static=True)

    def __init__(self):
        self.stem = ()
        self.stages = ()

    def features(self, x):
        return x


def test_replace_head_preserves_backbone(tiny_vision_transformer):
    key = jr.PRNGKey(0)
    new_head = eqft.LinearHead(4, 5, key=key)

    replaced = eqft.replace_head(tiny_vision_transformer, new_head)

    assert_tree_allclose(replaced.head, new_head)
    assert replaced.head(jnp.ones((4,))).shape == (5,)
    assert_tree_allclose(replaced.patch_embed, tiny_vision_transformer.patch_embed)
    assert_tree_allclose(replaced.blocks, tiny_vision_transformer.blocks)


def test_replace_head_validates_input_features(tiny_vision_transformer):
    with pytest.raises(ValueError, match="input-feature mismatch"):
        eqft.replace_head(
            tiny_vision_transformer,
            eqft.LinearHead(5, 3, key=jr.PRNGKey(0)),
        )


def test_replace_head_can_preserve_old_head_metadata(tiny_vision_transformer):
    replaced = eqft.replace_head(
        tiny_vision_transformer,
        eqft.LinearHead(4, 3, key=jr.PRNGKey(0)),
        preserve_old_head_metadata=True,
    )

    assert replaced.head.old_head_metadata["class_name"] == "Linear"
    assert replaced.head.old_head_metadata["in_features"] == 4
    assert replaced.head.old_head_metadata["out_features"] == 2
    assert replaced.head(jnp.ones((4,))).shape == (3,)


def test_extract_features_pool_cls_and_mean_patch(tiny_vision_transformer):
    x = jnp.ones((2, 3))

    cls = eqft.extract_features(tiny_vision_transformer, x, pool="cls")
    mean_patch = eqft.extract_features(tiny_vision_transformer, x, pool="mean_patch")

    assert cls.shape == (4,)
    assert mean_patch.shape == (4,)


def test_extract_features_auto_pool_audio_uses_mean_frame(tiny_ast_like_encoder):
    x = jnp.ones((2, 6))

    auto = eqft.extract_features(tiny_ast_like_encoder, x, pool="auto")
    mean_frame = eqft.extract_features(tiny_ast_like_encoder, x, pool="mean_frame")

    assert jnp.allclose(auto, mean_frame)


def test_extract_features_auto_pool_text_uses_mean_token(tiny_text_encoder):
    token_ids = jnp.asarray([0, 1, 2])

    auto = eqft.extract_features(tiny_text_encoder, token_ids, pool="auto")
    mean_token = eqft.extract_features(tiny_text_encoder, token_ids, pool="mean_token")

    assert jnp.allclose(auto, mean_token)


def test_extract_features_auto_pool_conv_features_uses_global_average():
    x = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 2, 2)

    pooled = eqft.extract_features(ConvFeatureModel(), x, pool="auto")

    assert jnp.array_equal(pooled, jnp.mean(x, axis=(1, 2)))


def test_extract_features_drops_optional_key_for_plain_features():
    features = eqft.extract_features(
        NoKwargFeatureModel(),
        jnp.ones((4,)),
        pool="cls",
        key=jr.PRNGKey(0),
    )

    assert features.shape == (4,)


def test_linear_probe_trainable_only_head(tiny_vision_transformer):
    key = jr.PRNGKey(1)
    probe = eqft.make_linear_probe(
        tiny_vision_transformer,
        in_features=4,
        out_features=3,
        key=key,
        pool="cls",
    )
    plan = eqft.prepare_finetune(
        probe,
        trainable=eqft.TrainableSpec(mode="head"),
    )

    assert isinstance(probe.backbone.head, eqft.IdentityHead)
    assert plan.trainable.head.linear.weight is not None
    assert plan.trainable.backbone.patch_embed.proj.weight is None
    assert plan.trainable.backbone.blocks[0].attn.qkv.weight is None
    assert plan.report.trainable_params == 15


def test_feature_extractor_filter_jit(tiny_vision_transformer):
    extractor = eqft.FeatureExtractor(tiny_vision_transformer, pool="cls")
    x = jnp.ones((2, 3))

    features = eqx.filter_jit(extractor)(x)

    assert features.shape == (4,)
