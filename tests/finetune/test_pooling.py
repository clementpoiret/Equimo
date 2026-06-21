"""Fine-tuning pooling tests."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


def test_pool_cls_shape():
    tokens = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
    pooled = eqft.CLSPool()(tokens)

    assert pooled.shape == (4,)
    assert jnp.array_equal(pooled, tokens[0])


def test_pool_mean_patch_excludes_cls():
    tokens = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
    pooled = eqft.MeanPatchPool(num_prefix_tokens=1)(tokens)

    assert pooled.shape == (4,)
    assert jnp.array_equal(pooled, jnp.mean(tokens[1:], axis=0))


def test_pool_mean_token_mask():
    tokens = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    mask = jnp.array([1, 1, 0])
    pooled = eqft.MeanTokenPool()(tokens, mask=mask)

    assert jnp.array_equal(pooled, jnp.mean(tokens[:2], axis=0))


def test_pool_attention_policy_uses_key():
    tokens = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)

    pooled = eqft.pool_features(tokens, pool="attention", key=jr.PRNGKey(0))

    assert pooled.shape == (4,)


def test_pool_attention_policy_requires_key():
    tokens = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)

    with pytest.raises(ValueError, match="requires a PRNG key"):
        eqft.pool_features(tokens, pool="attention")
