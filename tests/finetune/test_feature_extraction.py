"""Feature extraction and head replacement tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft

from fixtures import assert_tree_allclose


def test_replace_head_preserves_backbone(tiny_vision_transformer):
    key = jr.PRNGKey(0)
    new_head = eqft.LinearHead(4, 5, key=key)

    replaced = eqft.replace_head(tiny_vision_transformer, new_head)

    assert_tree_allclose(replaced.head, new_head)
    assert replaced.head(jnp.ones((4,))).shape == (5,)
    assert_tree_allclose(replaced.patch_embed, tiny_vision_transformer.patch_embed)
    assert_tree_allclose(replaced.blocks, tiny_vision_transformer.blocks)


def test_extract_features_pool_cls_and_mean_patch(tiny_vision_transformer):
    x = jnp.ones((2, 3))

    cls = eqft.extract_features(tiny_vision_transformer, x, pool="cls")
    mean_patch = eqft.extract_features(tiny_vision_transformer, x, pool="mean_patch")

    assert cls.shape == (4,)
    assert mean_patch.shape == (4,)


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
