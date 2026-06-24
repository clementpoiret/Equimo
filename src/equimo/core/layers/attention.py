# ty: ignore[unknown-argument]
# ty: ignore[invalid-assignment]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[call-non-callable]
from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core.layers.activation import get_act
from equimo.core.layers.dropout import DropPathAdd
from equimo.core.layers.ffn import get_ffn
from equimo.core.layers.norm import LayerScale, get_norm

_ATTN_REGISTRY: dict[str, type[eqx.Module]] = {}
_ATTN_BLOCK_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_attn(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a modality-neutral attention module."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()
        if registry_name in _ATTN_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_ATTN_REGISTRY[registry_name]}."
            )

        _ATTN_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_attn(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a modality-neutral attention class."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _ATTN_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_ATTN_REGISTRY.keys())}"
        )
    return _ATTN_REGISTRY[module_lower]


def register_attn_block(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a modality-neutral transformer block."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()
        if registry_name in _ATTN_BLOCK_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_ATTN_BLOCK_REGISTRY[registry_name]}."
            )

        _ATTN_BLOCK_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_attn_block(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a modality-neutral transformer block class."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _ATTN_BLOCK_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_ATTN_BLOCK_REGISTRY.keys())}"
        )
    return _ATTN_BLOCK_REGISTRY[module_lower]


def rope_rotate_half(x: jax.Array) -> jax.Array:
    """Rotate the last-dimension halves used by rotary embeddings."""

    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def rope_apply(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    """Apply precomputed rotary sine/cosine factors to an array."""

    return (x * cos) + (rope_rotate_half(x) * sin)


def rope_apply_qk_last_hw(
    q: jax.Array,
    k: jax.Array,
    sin: jax.Array,
    cos: jax.Array,
    prefix: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Apply RoPE to the tail tokens while preserving optional prefix tokens."""
    hw = sin.shape[-2]
    n = q.shape[-2]
    if prefix is None:
        prefix = n - hw
    if prefix < 0:
        raise ValueError(f"Sequence length N={n} smaller than HW={hw}.")

    q_dtype, k_dtype = q.dtype, k.dtype
    rope_dtype = sin.dtype
    q = q.astype(rope_dtype)
    k = k.astype(rope_dtype)

    sin_b = sin[None, :, :]
    cos_b = cos[None, :, :]
    q_prefix, q_tail = jnp.split(q, [prefix], axis=-2)
    k_prefix, k_tail = jnp.split(k, [prefix], axis=-2)
    q_tail = rope_apply(q_tail, sin_b, cos_b)
    k_tail = rope_apply(k_tail, sin_b, cos_b)
    return (
        jnp.concatenate([q_prefix, q_tail], axis=-2).astype(q_dtype),
        jnp.concatenate([k_prefix, k_tail], axis=-2).astype(k_dtype),
    )


@register_attn()
class Attention(eqx.Module):
    """Multi-head self attention for sequence tensors of shape ``(seq, dim)``."""

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    q_norm: eqx.Module
    k_norm: eqx.Module
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: str | type[eqx.Module] = "layernorm",
        eps: float = 1e-5,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        key_qkv, key_proj = jr.split(key, 2)
        norm_layer = get_norm(norm_layer)
        self.qkv = eqx.nn.Linear(dim, dim * 3, use_bias=qkv_bias, key=key_qkv)
        self.proj = eqx.nn.Linear(dim, dim, use_bias=proj_bias, key=key_proj)
        self.q_norm = (
            norm_layer(self.head_dim, eps=eps) if qk_norm else eqx.nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, eps=eps) if qk_norm else eqx.nn.Identity()
        )
        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        mask: Optional[Float[Array, ""]] = None,
        rope_sincos: Optional[Tuple[jax.Array, jax.Array]] = None,
    ) -> Float[Array, "seqlen dim"]:
        key1, key2 = jr.split(key, 2)

        qkv = jax.vmap(self.qkv)(x)
        qkv = rearrange(qkv, "s (n h d) -> n h s d", n=3, h=self.num_heads)
        q, k, v = qkv
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)

        if rope_sincos is not None:
            sin, cos = rope_sincos
            if sin.shape[-1] != self.head_dim or cos.shape[-1] != self.head_dim:
                raise ValueError(
                    f"RoPE sin/cos last dim ({sin.shape[-1]}) must equal "
                    f"head_dim ({self.head_dim})."
                )
            q, k = rope_apply_qk_last_hw(q, k, sin, cos)

        attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn = jnp.where(
                mask == 0, jnp.finfo(jnp.float32).min, attn.astype(jnp.float32)
            )
        else:
            attn = attn.astype(jnp.float32)
        attn = jax.nn.softmax(attn, axis=-1).astype(x.dtype)
        attn = self.attn_drop(attn, inference=inference, key=key1)

        x = jnp.einsum("hqk,hkd->hqd", attn, v)
        x = rearrange(x, "h s d -> s (h d)")
        x = jax.vmap(self.proj)(x)
        return self.proj_drop(x, inference=inference, key=key2)


@register_attn_block()
class AttentionBlock(eqx.Module):
    """Pre-norm transformer block for sequence tensors."""

    prenorm: eqx.Module
    postnorm: eqx.Module
    norm: eqx.Module
    ls1: LayerScale | eqx.nn.Identity
    ls2: LayerScale | eqx.nn.Identity
    attn: eqx.Module
    mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 4.0,
        drop_path: float | List[float] = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: str | Callable = "gelu",
        attn_layer: str | type[eqx.Module] = "attention",
        ffn_layer: str | type[eqx.Module] = "mlp",
        ffn_bias: bool = True,
        ffn_norm: bool = False,
        ffn_kwargs: dict = {},
        norm_layer: str | type[eqx.Module] = "layernorm",
        post_attention_norm: bool = False,
        init_values: float | None = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        key_attn, key_mlp = jr.split(key, 2)
        act_layer = get_act(act_layer)
        attn_layer = get_attn(attn_layer)
        ffn_layer = get_ffn(ffn_layer)
        norm_layer = get_norm(norm_layer)

        if isinstance(drop_path, list):
            if len(drop_path) == 1:
                dr1 = dr2 = float(drop_path[0])
            elif len(drop_path) == 2:
                dr1, dr2 = float(drop_path[0]), float(drop_path[1])
            else:
                raise AssertionError(
                    f"`drop_path` needs 1 or 2 elements, got {len(drop_path)}."
                )
        else:
            dr1 = dr2 = float(drop_path)

        self.prenorm = norm_layer(dim, eps=eps)
        self.postnorm = (
            norm_layer(dim, eps=eps) if post_attention_norm else eqx.nn.Identity()
        )
        self.norm = norm_layer(dim, eps=eps)
        self.attn = attn_layer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            eps=eps,
            key=key_attn,
        )
        self.mlp = ffn_layer(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if ffn_norm else None,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            eps=eps,
            key=key_mlp,
            **ffn_kwargs,
        )
        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)
        self.ls1 = (
            LayerScale(dim, axis=1, init_values=init_values)
            if init_values is not None
            else eqx.nn.Identity()
        )
        self.ls2 = (
            LayerScale(dim, axis=1, init_values=init_values)
            if init_values is not None
            else eqx.nn.Identity()
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        key_attn, key_mlp, key_dr1, key_dr2 = jr.split(key, 4)
        mask = kwargs.get("mask")
        extra_kwargs = {"mask": mask} if mask is not None else {}
        attn_kwargs = (
            extra_kwargs | {"rope_sincos": kwargs["rope_sincos"]}
            if kwargs.get("rope_sincos") is not None
            else extra_kwargs
        )
        x = self.drop_path1(
            x,
            self.ls1(
                jax.vmap(self.postnorm)(
                    self.attn(
                        jax.vmap(self.prenorm)(x),
                        inference=inference,
                        key=key_attn,
                        **attn_kwargs,
                    )
                )
            ),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.ls2(
                self.mlp(
                    jax.vmap(self.norm)(x),
                    inference=inference,
                    key=key_mlp,
                    **extra_kwargs,
                )
            ),
            inference=inference,
            key=key_dr2,
        )
        return x
