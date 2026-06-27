import jax.numpy as jnp
import pytest

from equimo.utils import pool_sd


def test_pool_sd_cls_patch_mean_excludes_prefix_tokens():
    tokens = jnp.arange(24, dtype=jnp.float32).reshape(6, 4)

    pooled = pool_sd(tokens, pool_type="cls_patch_mean", num_prefix_tokens=2)

    expected = jnp.concatenate([tokens[0], jnp.mean(tokens[2:], axis=0)], axis=0)
    assert pooled.shape == (8,)
    assert jnp.array_equal(pooled, expected)


def test_pool_sd_cls_patch_mean_requires_prefix_token():
    tokens = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

    with pytest.raises(ValueError, match="requires at least one prefix token"):
        pool_sd(tokens, pool_type="cls_patch_mean", num_prefix_tokens=0)
