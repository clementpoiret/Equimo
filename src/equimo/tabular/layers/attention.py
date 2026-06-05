from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from .mlp import Mlp, _call_mlp
from .registry import _register_module, _registry_name, _resolve_from_registry

_ATTN_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_attn(
    name: str | None = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a tabular attention layer class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        registry_name = _registry_name(cls, name)
        return _register_module(
            _ATTN_REGISTRY,
            cls,
            registry_name,
            force,
            "tabular attention",
            add_to_layer_registry=True,
        )

    return decorator


def get_attn(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a tabular attention layer class."""
    return _resolve_from_registry(module, _ATTN_REGISTRY, "tabular attention")


def _to_heads(
    proj: eqx.nn.Linear,
    x: Float[Array, "seqlen dim"],
    num_heads: int,
    head_dim: int,
) -> Float[Array, "heads seqlen head_dim"]:
    out = jax.vmap(proj)(x)
    return out.reshape(x.shape[0], num_heads, head_dim).transpose(1, 0, 2)


def _scaled_dot_product_attention(
    q: Float[Array, "heads q_len head_dim"],
    k: Float[Array, "kv_heads kv_len head_dim"],
    v: Float[Array, "kv_heads kv_len head_dim"],
    head_dim: int,
    scaling: eqx.Module | None = None,
    n: Array | int | None = None,
) -> Float[Array, "heads q_len head_dim"]:
    if scaling is not None:
        q = scaling(q, n)
    if k.shape[0] != q.shape[0]:
        k = jnp.repeat(k, q.shape[0] // k.shape[0], axis=0)
        v = jnp.repeat(v, q.shape[0] // v.shape[0], axis=0)
    scores = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(head_dim)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hqk,hkd->hqd", attn, v)


@register_attn()
class SoftmaxScaling(eqx.Module):
    """Input-dependent query scaling for tabular attention."""

    base_mlp: Mlp
    query_mlp: Mlp
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_dim: int = 64,
        *,
        key: PRNGKeyArray,
        act_layer: str | Callable = "exactgelu",
    ) -> None:
        key_base, key_query = jr.split(key, 2)
        self.base_mlp = Mlp(
            1,
            hidden_dim=hidden_dim,
            out_dim=num_heads * head_dim,
            act_layer=act_layer,
            dropout_rate=0.0,
            norm_layer=None,
            key=key_base,
        )
        self.query_mlp = Mlp(
            head_dim,
            hidden_dim=hidden_dim,
            out_dim=head_dim,
            act_layer=act_layer,
            dropout_rate=0.0,
            norm_layer=None,
            key=key_query,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(
        self,
        q: Float[Array, "heads seqlen head_dim"],
        n: Array | int,
    ) -> Float[Array, "heads seqlen head_dim"]:
        logn = jnp.log(jnp.maximum(jnp.asarray(n, q.dtype), 1.0)).reshape(1)
        base = _call_mlp(
            self.base_mlp, logn, inference=True
        ).reshape(self.num_heads, 1, self.head_dim)
        modulation = 1 + jnp.tanh(_call_mlp(self.query_mlp, q, inference=True))
        return q * base * modulation


@register_attn()
class Attention(eqx.Module):
    """Bias-free multi-head attention for tabular token sequences."""

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        head_dim: int | None = None,
    ) -> None:
        key_q, key_k, key_v, key_out = jr.split(key, 4)
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.dim = dim
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads.")
        inner_dim = num_heads * self.head_dim
        self.q_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_q
        )
        self.k_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_k
        )
        self.v_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_v
        )
        self.proj = eqx.nn.Linear(
            inner_dim, dim, use_bias=False, key=key_out
        )
        self.num_heads = num_heads

    def cross(
        self,
        q_tokens: Float[Array, "q_len dim"],
        kv_tokens: Float[Array, "kv_len dim"],
        rope: eqx.Module | None = None,
    ) -> Float[Array, "q_len dim"]:
        q = _to_heads(self.q_proj, q_tokens, self.num_heads, self.head_dim)
        k = _to_heads(self.k_proj, kv_tokens, self.num_heads, self.head_dim)
        v = _to_heads(self.v_proj, kv_tokens, self.num_heads, self.head_dim)
        if rope is not None:
            q = jax.vmap(rope)(q)
            k = jax.vmap(rope)(k)
        out = _scaled_dot_product_attention(q, k, v, self.head_dim)
        out = out.transpose(1, 0, 2).reshape(q_tokens.shape[0], -1)
        return jax.vmap(self.proj)(out)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
        rope: eqx.Module | None = None,
    ) -> Float[Array, "seqlen dim"]:
        return self.cross(x, x, rope)


@register_attn()
class CrossAttention(eqx.Module):
    """Cross-attention with optional query scaling."""

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    proj: eqx.nn.Linear
    softmax_scaling: SoftmaxScaling | None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        softmax_scaling: SoftmaxScaling | None = None,
        head_dim: int | None = None,
    ) -> None:
        key_q, key_k, key_v, key_out = jr.split(key, 4)
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.dim = dim
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads.")
        inner_dim = num_heads * self.head_dim
        self.q_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_q
        )
        self.k_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_k
        )
        self.v_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_v
        )
        self.proj = eqx.nn.Linear(
            inner_dim, dim, use_bias=False, key=key_out
        )
        self.softmax_scaling = softmax_scaling
        self.num_heads = num_heads

    def __call__(
        self,
        q_tokens: Float[Array, "q_len dim"],
        kv_tokens: Float[Array, "kv_len dim"],
    ) -> Float[Array, "q_len dim"]:
        q = _to_heads(self.q_proj, q_tokens, self.num_heads, self.head_dim)
        k = _to_heads(self.k_proj, kv_tokens, self.num_heads, self.head_dim)
        v = _to_heads(self.v_proj, kv_tokens, self.num_heads, self.head_dim)
        out = _scaled_dot_product_attention(
            q, k, v, self.head_dim, self.softmax_scaling, kv_tokens.shape[0]
        )
        out = out.transpose(1, 0, 2).reshape(q_tokens.shape[0], -1)
        return jax.vmap(self.proj)(out)


@register_attn()
class InContextAttention(eqx.Module):
    """Self-attention whose keys/values are restricted to training rows."""

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_kv_heads_test: int | None = eqx.field(static=True)

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    proj: eqx.nn.Linear
    softmax_scaling: SoftmaxScaling | None

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        softmax_scaling: SoftmaxScaling | None = None,
        num_kv_heads_test: int | None = None,
        head_dim: int | None = None,
    ) -> None:
        key_q, key_k, key_v, key_out = jr.split(key, 4)
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.dim = dim
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads.")
        inner_dim = num_heads * self.head_dim
        self.q_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_q
        )
        self.k_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_k
        )
        self.v_proj = eqx.nn.Linear(
            dim, inner_dim, use_bias=False, key=key_v
        )
        self.proj = eqx.nn.Linear(
            inner_dim, dim, use_bias=False, key=key_out
        )
        self.softmax_scaling = softmax_scaling
        self.num_heads = num_heads
        self.num_kv_heads_test = num_kv_heads_test

    def __call__(
        self,
        x: Float[Array, "rows dim"],
        n_train: int,
    ) -> Float[Array, "rows dim"]:
        q = _to_heads(self.q_proj, x, self.num_heads, self.head_dim)
        k = _to_heads(self.k_proj, x, self.num_heads, self.head_dim)[
            :, :n_train
        ]
        v = _to_heads(self.v_proj, x, self.num_heads, self.head_dim)[
            :, :n_train
        ]

        if self.num_kv_heads_test is None:
            out = _scaled_dot_product_attention(
                q, k, v, self.head_dim, self.softmax_scaling, n_train
            )
        else:
            kv_heads = self.num_kv_heads_test
            out_train = _scaled_dot_product_attention(
                q[:, :n_train], k, v, self.head_dim, self.softmax_scaling, n_train
            )
            out_test = _scaled_dot_product_attention(
                q[:, n_train:],
                k[:kv_heads],
                v[:kv_heads],
                self.head_dim,
                self.softmax_scaling,
                n_train,
            )
            out = jnp.concatenate([out_train, out_test], axis=1)

        out = out.transpose(1, 0, 2).reshape(x.shape[0], -1)
        return jax.vmap(self.proj)(out)
