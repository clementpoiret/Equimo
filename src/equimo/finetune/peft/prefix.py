"""Prefix-tuning wrappers for supported attention modules."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..paths import key_path_to_path
from .base import get_path


@dataclass(frozen=True)
class PrefixConfig:
    """Configuration for prefix tuning."""

    num_prefix_tokens: int = 16
    depth: str = "deep"
    target: tuple[str, ...] = ("attention.k", "attention.v")
    prefix_projection: bool = True
    direct_kv: bool = False
    projection_hidden_dim: int | str = "model_dim"
    init_std: float = 0.02
    prefix_dropout: float = 0.0
    train_head: bool = True


class PrefixTunedModel(eqx.Module):
    """Store trainable prefix parameters next to a prefix-wrapped base model."""

    base: PyTree
    prefixes: tuple[jax.Array, ...]
    prefix_projections: tuple[PrefixProjection | None, ...]
    config: PrefixConfig = eqx.field(static=True)

    def __init__(
        self,
        base: PyTree,
        prefixes: tuple[jax.Array, ...],
        config: PrefixConfig,
        prefix_projections: tuple[PrefixProjection | None, ...] | None = None,
    ):
        _validate_prefix_config(config)
        if not prefixes:
            raise ValueError("PrefixTunedModel requires at least one prefix tensor.")
        if prefix_projections is None:
            prefix_projections = (None,) * len(prefixes)
        if len(prefix_projections) != len(prefixes):
            raise ValueError(
                "PrefixTunedModel requires one projection entry per prefix tensor."
            )
        self.base = _ensure_prefix_wrapped(
            base,
            prefixes,
            prefix_projections,
            config,
        )
        self.prefixes = prefixes
        self.prefix_projections = prefix_projections
        self.config = config

    def features(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        **kwargs,
    ):
        base = _sync_prefixes(self.base, self.prefixes, self.prefix_projections)
        if not hasattr(base, "features"):
            raise ValueError(
                "PrefixTunedModel requires the base model to expose features()."
            )
        return _call_with_optional_key(
            base.features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        **kwargs,
    ):
        base = _sync_prefixes(self.base, self.prefixes, self.prefix_projections)
        return _call_with_optional_key(
            base,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )


class PrefixAttention(eqx.Module):
    """Attention wrapper that prepends trainable K/V prefix states."""

    base: eqx.Module
    state: jax.Array
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    prefix_dropout: float = eqx.field(static=True)

    def __init__(
        self,
        base: eqx.Module,
        state: jax.Array,
        *,
        prefix_dropout: float = 0.0,
    ):
        self.base = base
        self.state = state
        self.num_heads = int(state.shape[1])
        self.head_dim = int(state.shape[3])
        self.prefix_dropout = prefix_dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = None,
        mask: jax.Array | None = None,
        rope_sincos=None,
    ) -> jax.Array:
        key_prefix, key1, key2 = _split_three(key)
        q, k, v = _project_qkv(self.base, x, self.num_heads, self.head_dim)
        q = _apply_nested_norm(getattr(self.base, "q_norm", None), q)
        k = _apply_nested_norm(getattr(self.base, "k_norm", None), k)

        if rope_sincos is not None:
            q, k = _apply_rope(q, k, rope_sincos)

        prefix_k, prefix_v = self.state.astype(x.dtype)
        if self.prefix_dropout > 0.0 and not inference:
            if key_prefix is None:
                raise ValueError(
                    "A PRNG key is required when prefix dropout is active."
                )
            key_k, key_v = jr.split(key_prefix, 2)
            prefix_k = _dropout(prefix_k, self.prefix_dropout, key_k)
            prefix_v = _dropout(prefix_v, self.prefix_dropout, key_v)
        k = jnp.concatenate([prefix_k, k], axis=1)
        v = jnp.concatenate([prefix_v, v], axis=1)
        mask = _extend_mask(mask, self.state.shape[2])

        attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn = jnp.where(
                mask == 0,
                jnp.finfo(jnp.float32).min,
                attn.astype(jnp.float32),
            )
        else:
            attn = attn.astype(jnp.float32)
        attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)
        attn = _call_dropout(
            getattr(self.base, "attn_drop", None), attn, inference, key1
        )

        y = jnp.einsum("hqk,hkd->hqd", attn, v)
        y = jnp.transpose(y, (1, 0, 2)).reshape((x.shape[0], -1))
        y = jax.vmap(self.base.proj)(y)
        return _call_dropout(getattr(self.base, "proj_drop", None), y, inference, key2)


class PrefixProjection(eqx.Module):
    """Trainable MLP that projects prefix tokens into K/V prefix states."""

    down: eqx.nn.Linear
    up: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        key: jax.Array,
        init_std: float = 0.02,
        down: eqx.nn.Linear | None = None,
        up: eqx.nn.Linear | None = None,
    ):
        down_key, up_key = jr.split(key, 2)
        self.down = (
            _init_projection_linear(
                eqx.nn.Linear(dim, hidden_dim, key=down_key),
                down_key,
                init_std,
            )
            if down is None
            else down
        )
        self.up = (
            _init_projection_linear(
                eqx.nn.Linear(
                    hidden_dim,
                    2 * num_heads * head_dim,
                    key=up_key,
                ),
                up_key,
                init_std,
            )
            if up is None
            else up
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, prefix: jax.Array) -> jax.Array:
        hidden = jax.vmap(self.down)(prefix)
        state = jax.vmap(self.up)(jnp.tanh(hidden))
        state = state.reshape((prefix.shape[0], 2, self.num_heads, self.head_dim))
        return jnp.transpose(state, (1, 2, 0, 3))


def apply_prefixes(
    model: PyTree,
    config: PrefixConfig | None = None,
    *,
    key: jax.Array,
) -> PrefixTunedModel:
    """Attach trainable prefix tensors to supported attention modules."""

    config = PrefixConfig() if config is None else config
    _validate_prefix_config(config)
    paths = _prefix_attention_paths(model)
    if config.depth == "shallow":
        paths = paths[:1]
    if not paths:
        raise ValueError(
            "Prefix tuning requires attention modules with qkv/proj projections."
        )

    keys = jr.split(key, len(paths))
    prefixes = tuple(
        _init_prefix(
            get_path(model, path), config.num_prefix_tokens, config.init_std, subkey
        )
        for path, subkey in zip(paths, keys, strict=True)
    )
    projection_keys = jr.split(jr.fold_in(key, 1), len(paths))
    projections = tuple(
        _init_prefix_projection(
            get_path(model, path),
            config,
            subkey,
        )
        for path, subkey in zip(paths, projection_keys, strict=True)
    )
    updated = _wrap_prefix_attentions(model, paths, prefixes, projections, config)
    return PrefixTunedModel(updated, prefixes, config, projections)


def strip_prefixes(model: PyTree) -> PyTree:
    """Replace prefix-attention wrappers with their base attention modules."""

    stripped = model
    for path, wrapper in iter_prefix_attentions(stripped):
        stripped = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p), stripped, wrapper.base
        )
    return stripped


def iter_prefix_attentions(model: PyTree) -> tuple[tuple[Path, PrefixAttention], ...]:
    """Return path/module pairs for prefix-attention wrappers."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, PrefixAttention),
        )
        if isinstance(leaf, PrefixAttention)
    )


def _ensure_prefix_wrapped(
    base: PyTree,
    prefixes: tuple[jax.Array, ...],
    projections: tuple[PrefixProjection | None, ...],
    config: PrefixConfig,
) -> PyTree:
    wrappers = iter_prefix_attentions(base)
    if wrappers:
        if len(wrappers) != len(prefixes):
            raise ValueError(
                "Prefix bundle/module mismatch: "
                f"{len(prefixes)} prefixes for {len(wrappers)} wrapped attentions."
            )
        return _sync_prefixes(base, prefixes, projections)

    paths = _prefix_attention_paths(base)
    if len(paths) != len(prefixes):
        raise ValueError(
            "Prefix tuning requires attention modules with qkv/proj projections."
        )
    return _wrap_prefix_attentions(base, paths, prefixes, projections, config)


def _wrap_prefix_attentions(
    model: PyTree,
    paths: tuple[Path, ...],
    prefixes: tuple[jax.Array, ...],
    projections: tuple[PrefixProjection | None, ...],
    config: PrefixConfig,
) -> PyTree:
    updated = model
    for path, prefix, projection in zip(paths, prefixes, projections, strict=True):
        module = get_path(updated, path)
        state = _prefix_to_state(prefix, module, projection)
        if isinstance(module, PrefixAttention):
            wrapper = eqx.tree_at(lambda item: item.state, module, state)
        else:
            wrapper = PrefixAttention(
                module,
                state,
                prefix_dropout=config.prefix_dropout,
            )
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


def _sync_prefixes(
    base: PyTree,
    prefixes: tuple[jax.Array, ...],
    projections: tuple[PrefixProjection | None, ...],
) -> PyTree:
    updated = base
    wrappers = iter_prefix_attentions(updated)
    if len(wrappers) != len(prefixes):
        raise ValueError(
            "Prefix bundle/module mismatch: "
            f"{len(prefixes)} prefixes for {len(wrappers)} wrapped attentions."
        )
    for (path, wrapper), prefix, projection in zip(
        wrappers,
        prefixes,
        projections,
        strict=True,
    ):
        state = _prefix_to_state(prefix, wrapper, projection)
        updated = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p).state,
            updated,
            state.astype(wrapper.state.dtype),
        )
    return updated


def _prefix_attention_paths(model: PyTree) -> tuple[Path, ...]:
    paths = tuple(
        key_path_to_path(key_path)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=_is_supported_attention,
        )
        if _is_supported_attention(leaf)
    )
    return paths


def _is_supported_attention(module) -> bool:
    if isinstance(module, PrefixAttention):
        return True
    return (
        hasattr(module, "qkv")
        and hasattr(module, "proj")
        and isinstance(module.qkv, eqx.nn.Linear)
        and isinstance(module.proj, eqx.nn.Linear)
        and module.qkv.out_features % 3 == 0
    )


def _init_prefix(
    module,
    num_tokens: int,
    init_std: float,
    key: jax.Array,
) -> jax.Array:
    return (
        jr.normal(
            key,
            (num_tokens, module.qkv.in_features),
            dtype=module.qkv.weight.dtype,
        )
        * init_std
    )


def _init_prefix_projection(
    module,
    config: PrefixConfig,
    key: jax.Array,
) -> PrefixProjection | None:
    if not config.prefix_projection:
        return None
    num_heads, head_dim = _attention_shape(module)
    hidden_dim = (
        module.qkv.in_features
        if config.projection_hidden_dim == "model_dim"
        else int(config.projection_hidden_dim)
    )
    return PrefixProjection(
        module.qkv.in_features,
        hidden_dim,
        num_heads,
        head_dim,
        key=key,
        init_std=config.init_std,
    )


def _init_projection_linear(
    linear: eqx.nn.Linear,
    key: jax.Array,
    init_std: float,
) -> eqx.nn.Linear:
    weight = jr.normal(key, linear.weight.shape, dtype=linear.weight.dtype) * init_std
    linear = eqx.tree_at(lambda layer: layer.weight, linear, weight)
    if linear.bias is not None:
        linear = eqx.tree_at(
            lambda layer: layer.bias, linear, jnp.zeros_like(linear.bias)
        )
    return linear


def _attention_shape(module) -> tuple[int, int]:
    if isinstance(module, PrefixAttention):
        return module.num_heads, module.head_dim
    qkv_dim = module.qkv.out_features // 3
    num_heads = int(getattr(module, "num_heads", 1))
    head_dim = int(getattr(module, "head_dim", qkv_dim // num_heads))
    if num_heads * head_dim != qkv_dim:
        raise ValueError(
            f"Unsupported attention projection shape: qkv output {module.qkv.out_features}."
        )
    return num_heads, head_dim


def _prefix_to_state(
    prefix: jax.Array,
    module,
    projection: PrefixProjection | None = None,
) -> jax.Array:
    if prefix.ndim == 4:
        return prefix
    if prefix.ndim != 2:
        raise ValueError("Prefix tensors must have shape (tokens, dim).")
    if projection is not None:
        return projection(prefix)
    num_heads, head_dim = _attention_shape(module)
    if prefix.shape[-1] != num_heads * head_dim:
        raise ValueError(
            "Prefix tensor feature dimension must match attention projection dimension."
        )
    tokens = jnp.transpose(
        prefix.reshape((prefix.shape[0], num_heads, head_dim)),
        (1, 0, 2),
    )
    return jnp.stack([tokens, tokens], axis=0)


def _project_qkv(module, x: jax.Array, num_heads: int, head_dim: int):
    qkv = jax.vmap(module.qkv)(x)
    qkv = qkv.reshape((x.shape[0], 3, num_heads, head_dim))
    qkv = jnp.transpose(qkv, (1, 2, 0, 3))
    return qkv[0], qkv[1], qkv[2]


def _apply_nested_norm(norm, x: jax.Array) -> jax.Array:
    if norm is None:
        return x
    return jax.vmap(jax.vmap(norm))(x)


def _apply_rope(q: jax.Array, k: jax.Array, rope_sincos):
    try:
        from equimo.vision.layers.attention import rope_apply_qk_last_hw
    except Exception:
        return q, k
    sin, cos = rope_sincos
    return rope_apply_qk_last_hw(q, k, sin, cos)


def _extend_mask(mask: jax.Array | None, num_prefix_tokens: int) -> jax.Array | None:
    if mask is None:
        return None
    prefix_shape = (*mask.shape[:-2], mask.shape[-2], num_prefix_tokens)
    prefix = jnp.ones(prefix_shape, dtype=mask.dtype)
    return jnp.concatenate([prefix, mask], axis=-1)


def _call_dropout(module, x: jax.Array, inference: bool | None, key: jax.Array | None):
    if module is None:
        return x
    return module(x, inference=inference, key=key)


def _validate_prefix_config(config: PrefixConfig) -> None:
    if config.depth not in {"shallow", "deep", "all"}:
        raise ValueError("PrefixConfig.depth must be 'shallow', 'deep', or 'all'.")
    if frozenset(config.target) != frozenset(("attention.k", "attention.v")):
        raise ValueError(
            "PrefixConfig.target currently supports only attention.k and attention.v."
        )
    if config.direct_kv:
        raise ValueError(
            "PrefixConfig.direct_kv=True is declared but not implemented; "
            "use projected prefix tokens instead."
        )


def _dropout(x: jax.Array, rate: float, key: jax.Array) -> jax.Array:
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def _call_with_optional_key(fn, *args, key, inference, **kwargs):
    call_kwargs = dict(kwargs)
    if key is not None:
        call_kwargs["key"] = key
    if inference is not None:
        call_kwargs["inference"] = inference
    try:
        return fn(*args, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        if key is None:
            call_kwargs.pop("key", None)
        return fn(*args, **call_kwargs)


def _split_three(
    key: jax.Array | None,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None]:
    if key is None:
        return None, None, None
    key_a, key_b, key_c = jr.split(key, 3)
    return key_a, key_b, key_c


__all__ = (
    "PrefixAttention",
    "PrefixConfig",
    "PrefixProjection",
    "PrefixTunedModel",
    "apply_prefixes",
    "iter_prefix_attentions",
    "strip_prefixes",
)
