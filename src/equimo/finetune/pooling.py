"""Pooling policies for fine-tuning feature extraction."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from equimo.utils import pool_sd


PoolName = Literal[
    "auto",
    "none",
    "cls",
    "cls_patch_mean",
    "global_avg",
    "mean_token",
    "mean_patch",
    "mean_frame",
    "attention",
    "gem",
    "last_token",
]


class CLSPool(eqx.Module):
    """Return the CLS/prefix token at ``index``."""

    index: int = eqx.field(static=True)

    def __init__(self, index: int = 0):
        self.index = index

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        return x[self.index]


class MeanTokenPool(eqx.Module):
    """Mean-pool token or frame features."""

    axis: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, axis: int = 0, eps: float = 1e-12):
        self.axis = axis
        self.eps = eps

    def __call__(self, x: jax.Array, *, mask: jax.Array | None = None) -> jax.Array:
        if mask is None:
            return jnp.mean(x, axis=self.axis)

        weights = mask.astype(x.dtype)
        while weights.ndim < x.ndim:
            weights = jnp.expand_dims(weights, axis=-1)
        total = jnp.sum(weights, axis=self.axis)
        return jnp.sum(x * weights, axis=self.axis) / jnp.maximum(total, self.eps)


class MeanPatchPool(eqx.Module):
    """Mean-pool patch tokens while excluding prefix/prompt tokens."""

    num_prefix_tokens: int = eqx.field(static=True)
    num_prompt_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        num_prefix_tokens: int = 1,
        *,
        num_prompt_tokens: int = 0,
    ):
        self.num_prefix_tokens = num_prefix_tokens
        self.num_prompt_tokens = num_prompt_tokens

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        start = self.num_prefix_tokens + self.num_prompt_tokens
        return jnp.mean(x[start:], axis=0)


class CLSPatchMeanPool(eqx.Module):
    """Concatenate the CLS token with the mean over patch tokens."""

    num_prefix_tokens: int = eqx.field(static=True)
    num_prompt_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        num_prefix_tokens: int = 1,
        *,
        num_prompt_tokens: int = 0,
    ):
        self.num_prefix_tokens = num_prefix_tokens
        self.num_prompt_tokens = num_prompt_tokens

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        return pool_sd(
            x,
            pool_type="cls_patch_mean",
            num_prefix_tokens=self.num_prefix_tokens + self.num_prompt_tokens,
            reduce_include_prefix=False,
        )


class MeanFramePool(MeanTokenPool):
    """Alias module for audio/frame feature pooling."""


class AttentionPool(eqx.Module):
    """Single-query attention pooling over token features."""

    query: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, *, key: jax.Array, eps: float = 1e-12):
        self.query = jr.normal(key, (dim,), dtype=jnp.float32) / jnp.sqrt(dim)
        self.eps = eps

    def __call__(self, x: jax.Array, *, mask: jax.Array | None = None) -> jax.Array:
        scores = x @ self.query
        if mask is not None:
            scores = jnp.where(mask.astype(bool), scores, -jnp.inf)
        weights = jax.nn.softmax(scores, axis=0)
        weights = jnp.nan_to_num(weights)
        pooled = jnp.sum(x * weights[:, None], axis=0)
        return pooled / jnp.maximum(jnp.sum(weights), self.eps)


class GeMPool(eqx.Module):
    """Generalized mean pooling over a token/frame axis."""

    p: float = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, p: float = 3.0, axis: int = 0, eps: float = 1e-6):
        self.p = p
        self.axis = axis
        self.eps = eps

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        x = jnp.clip(x, min=self.eps)
        return jnp.mean(x**self.p, axis=self.axis) ** (1.0 / self.p)


class LastTokenPool(eqx.Module):
    """Return the last token, or the last valid token under ``mask``."""

    def __call__(self, x: jax.Array, *, mask: jax.Array | None = None) -> jax.Array:
        if mask is None:
            return x[-1]
        index = jnp.maximum(jnp.sum(mask.astype(jnp.int32)) - 1, 0)
        return x[index]


class TokenIndexPool(eqx.Module):
    """Return a token at a fixed index."""

    index: int = eqx.field(static=True)

    def __init__(self, index: int):
        self.index = index

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        return x[self.index]


class GlobalAveragePool(eqx.Module):
    """Mean-pool all non-channel axes of dense channel-first features."""

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        if x.ndim <= 1:
            return x
        if x.ndim == 2:
            return jnp.mean(x, axis=0)
        return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def pool_features(
    features: jax.Array | dict,
    pool: PoolName | eqx.Module | None = "auto",
    **kwargs,
) -> jax.Array | dict:
    """Apply a pooling policy to feature arrays or return dictionaries unchanged."""

    pool_kwargs = dict(kwargs)
    pool_key = pool_kwargs.pop("key", None)
    if pool is None or pool == "none":
        return features
    if isinstance(features, dict):
        return _pool_feature_dict(features, pool, key=pool_key, **pool_kwargs)
    if isinstance(pool, eqx.Module):
        return pool(features, **pool_kwargs)
    if pool == "attention":
        if pool_key is None:
            raise ValueError("pool='attention' requires a PRNG key.")
        return AttentionPool(features.shape[-1], key=pool_key)(
            features,
            **pool_kwargs,
        )
    return _pool_module(pool, features)(features, **pool_kwargs)


def _pool_feature_dict(
    features: dict,
    pool: PoolName | eqx.Module,
    **kwargs,
) -> jax.Array | dict:
    if pool in {"auto", "cls"} and "x_norm_cls_token" in features:
        return features["x_norm_cls_token"]
    if pool == "cls_patch_mean" and (
        "x_norm_cls_token" in features or "x_norm_patchtokens" in features
    ):
        cls_token = features.get("x_norm_cls_token")
        if cls_token is None or "x_norm_patchtokens" not in features:
            raise ValueError(
                "pool='cls_patch_mean' requires CLS and patch-token features."
            )
        return jnp.concatenate(
            [cls_token, jnp.mean(features["x_norm_patchtokens"], axis=0)],
            axis=0,
        )
    if pool == "mean_patch" and "x_norm_patchtokens" in features:
        return jnp.mean(features["x_norm_patchtokens"], axis=0)
    if pool == "mean_frame" and "x_norm_patchtokens" in features:
        return jnp.mean(features["x_norm_patchtokens"], axis=0)
    if pool == "auto" and "pooler_output" in features:
        return features["pooler_output"]
    if pool == "auto" and "last_hidden_state" in features:
        return pool_features(features["last_hidden_state"], "mean_token", **kwargs)
    if "x_prenorm" in features:
        return pool_features(features["x_prenorm"], pool, **kwargs)
    return features


def _pool_module(pool: PoolName, features: jax.Array) -> eqx.Module:
    if pool == "auto":
        if features.ndim <= 1:
            return IdentityPool()
        return CLSPool()
    if pool == "cls":
        return CLSPool()
    if pool == "cls_patch_mean":
        return CLSPatchMeanPool()
    if pool == "global_avg":
        return GlobalAveragePool()
    if pool == "mean_token":
        return MeanTokenPool()
    if pool == "mean_patch":
        return MeanPatchPool()
    if pool == "mean_frame":
        return MeanFramePool()
    if pool == "gem":
        return GeMPool()
    if pool == "last_token":
        return LastTokenPool()
    raise ValueError(f"Unsupported pool policy {pool!r}.")


class IdentityPool(eqx.Module):
    """Return features unchanged."""

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        return x


__all__ = (
    "AttentionPool",
    "CLSPatchMeanPool",
    "CLSPool",
    "GeMPool",
    "GlobalAveragePool",
    "IdentityPool",
    "LastTokenPool",
    "MeanFramePool",
    "MeanPatchPool",
    "MeanTokenPool",
    "PoolName",
    "TokenIndexPool",
    "pool_features",
)
