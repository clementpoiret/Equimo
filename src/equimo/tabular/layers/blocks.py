# ty: ignore[invalid-assignment]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[unknown-argument]
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core.layers.dropout import DropPathAdd
from equimo.core.layers.generic import BlockChunk
from equimo.core.layers.norm import get_norm

from .attention import (
    Attention,
    CrossAttention,
    InContextAttention,
    SoftmaxScaling,
)
from .mlp import Mlp, _call_mlp
from .registry import (
    _register_module,
    _registry_name,
    _resolve_from_registry,
    register_layer,
)

_ATTN_BLOCK_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_attn_block(
    name: str | None = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a tabular attention block class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        registry_name = _registry_name(cls, name)
        return _register_module(
            _ATTN_BLOCK_REGISTRY,
            cls,
            registry_name,
            force,
            "tabular attention block",
            add_to_layer_registry=True,
        )

    return decorator


def get_attn_block(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a tabular attention block class."""
    return _resolve_from_registry(
        module, _ATTN_BLOCK_REGISTRY, "tabular attention block"
    )


def _split_drop_path(drop_path: float | list[float]) -> tuple[float, float]:
    if isinstance(drop_path, list):
        if len(drop_path) == 1:
            return float(drop_path[0]), float(drop_path[0])
        if len(drop_path) == 2:
            return float(drop_path[0]), float(drop_path[1])
        raise AssertionError(
            f"`drop_path` needs 1 or 2 elements, got {len(drop_path)}."
        )
    return float(drop_path), float(drop_path)


def _split_optional_key(
    key: PRNGKeyArray | None,
    num: int,
) -> tuple[PRNGKeyArray | None, ...]:
    if key is None:
        return (None,) * num
    return tuple(jr.split(key, num))


@register_attn_block()
class CrossAttentionBlock(eqx.Module):
    """Pre-norm cross-attention block for tabular set processing."""

    attn: CrossAttention
    mlp: Mlp
    norm_q: eqx.Module
    norm_kv: eqx.Module
    norm_mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 2.0,
        drop_path: float | list[float] = 0.0,
        use_softmax_scaling: bool = False,
        scaling_mlp_hidden_dim: int = 64,
        act_layer: str | Callable = "exactgelu",
        norm_layer: str | type[eqx.Module] = "rmsnorm",
        eps: float = 1e-5,
    ) -> None:
        key_attn, key_mlp, key_scaling = jr.split(key, 3)
        norm_layer = get_norm(norm_layer)
        head_dim = dim // num_heads
        scaling = (
            SoftmaxScaling(
                num_heads,
                head_dim,
                scaling_mlp_hidden_dim,
                act_layer=act_layer,
                key=key_scaling,
            )
            if use_softmax_scaling
            else None
        )
        self.attn = CrossAttention(
            dim, num_heads, softmax_scaling=scaling, key=key_attn
        )
        self.mlp = Mlp(
            dim,
            hidden_dim=int(dim * mlp_ratio),
            bias=False,
            act_layer=act_layer,
            dropout_rate=0.0,
            norm_layer=None,
            key=key_mlp,
        )
        self.norm_q = norm_layer(dim, eps=eps)
        self.norm_kv = norm_layer(dim, eps=eps)
        self.norm_mlp = norm_layer(dim, eps=eps)
        dr1, dr2 = _split_drop_path(drop_path)
        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        q_tokens: Float[Array, "q_len dim"],
        kv_tokens: Float[Array, "kv_len dim"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "q_len dim"]:
        key_attn, key_mlp = _split_optional_key(key, 2)
        x = self.drop_path1(
            q_tokens,
            self.attn(
                jax.vmap(self.norm_q)(q_tokens),
                jax.vmap(self.norm_kv)(kv_tokens),
            ),
            inference=inference,
            key=key_attn,
        )
        return self.drop_path2(
            x,
            _call_mlp(
                self.mlp,
                jax.vmap(self.norm_mlp)(x),
                key=key_mlp,
                inference=inference,
            ),
            inference=inference,
            key=key_mlp,
        )


@register_attn_block()
class AttentionBlock(eqx.Module):
    """Pre-norm transformer block for tabular token sequences."""

    attn: Attention
    mlp: Mlp
    norm: eqx.Module
    norm_mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 2.0,
        drop_path: float | list[float] = 0.0,
        act_layer: str | Callable = "exactgelu",
        norm_layer: str | type[eqx.Module] = "rmsnorm",
        eps: float = 1e-5,
    ) -> None:
        key_attn, key_mlp = jr.split(key, 2)
        norm_layer = get_norm(norm_layer)
        self.attn = Attention(dim, num_heads, key=key_attn)
        self.mlp = Mlp(
            dim,
            hidden_dim=int(dim * mlp_ratio),
            bias=False,
            act_layer=act_layer,
            dropout_rate=0.0,
            norm_layer=None,
            key=key_mlp,
        )
        self.norm = norm_layer(dim, eps=eps)
        self.norm_mlp = norm_layer(dim, eps=eps)
        dr1, dr2 = _split_drop_path(drop_path)
        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
        rope: eqx.Module | None = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        key_attn, key_mlp = _split_optional_key(key, 2)
        x = self.drop_path1(
            x,
            self.attn(jax.vmap(self.norm)(x), rope=rope),
            inference=inference,
            key=key_attn,
        )
        return self.drop_path2(
            x,
            _call_mlp(
                self.mlp,
                jax.vmap(self.norm_mlp)(x),
                key=key_mlp,
                inference=inference,
            ),
            inference=inference,
            key=key_mlp,
        )

    def forward_cross(
        self,
        q_tokens: Float[Array, "q_len dim"],
        kv_tokens: Float[Array, "kv_len dim"],
        rope: eqx.Module | None = None,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> Float[Array, "q_len dim"]:
        key_attn, key_mlp = _split_optional_key(key, 2)
        x = self.drop_path1(
            q_tokens,
            self.attn.cross(
                jax.vmap(self.norm)(q_tokens),
                jax.vmap(self.norm)(kv_tokens),
                rope,
            ),
            inference=inference,
            key=key_attn,
        )
        return self.drop_path2(
            x,
            _call_mlp(
                self.mlp,
                jax.vmap(self.norm_mlp)(x),
                key=key_mlp,
                inference=inference,
            ),
            inference=inference,
            key=key_mlp,
        )


@register_attn_block()
class InContextAttentionBlock(eqx.Module):
    """Pre-norm block with keys/values restricted to training rows."""

    attn: InContextAttention
    mlp: Mlp
    norm: eqx.Module
    norm_mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 2.0,
        drop_path: float | list[float] = 0.0,
        use_softmax_scaling: bool = True,
        scaling_mlp_hidden_dim: int = 64,
        num_kv_heads_test: int | None = 1,
        act_layer: str | Callable = "exactgelu",
        norm_layer: str | type[eqx.Module] = "rmsnorm",
        eps: float = 1e-5,
    ) -> None:
        key_attn, key_mlp, key_scaling = jr.split(key, 3)
        norm_layer = get_norm(norm_layer)
        head_dim = dim // num_heads
        scaling = (
            SoftmaxScaling(
                num_heads,
                head_dim,
                scaling_mlp_hidden_dim,
                act_layer=act_layer,
                key=key_scaling,
            )
            if use_softmax_scaling
            else None
        )
        self.attn = InContextAttention(
            dim,
            num_heads,
            softmax_scaling=scaling,
            num_kv_heads_test=num_kv_heads_test,
            key=key_attn,
        )
        self.mlp = Mlp(
            dim,
            hidden_dim=int(dim * mlp_ratio),
            bias=False,
            act_layer=act_layer,
            dropout_rate=0.0,
            norm_layer=None,
            key=key_mlp,
        )
        self.norm = norm_layer(dim, eps=eps)
        self.norm_mlp = norm_layer(dim, eps=eps)
        dr1, dr2 = _split_drop_path(drop_path)
        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "rows dim"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
        n_train: int,
        **kwargs,
    ) -> Float[Array, "rows dim"]:
        key_attn, key_mlp = _split_optional_key(key, 2)
        x = self.drop_path1(
            x,
            self.attn(jax.vmap(self.norm)(x), n_train),
            inference=inference,
            key=key_attn,
        )
        return self.drop_path2(
            x,
            _call_mlp(
                self.mlp,
                jax.vmap(self.norm_mlp)(x),
                key=key_mlp,
                inference=inference,
            ),
            inference=inference,
            key=key_mlp,
        )


@register_attn_block()
class InducedAttentionBlock(eqx.Module):
    """Set Transformer-style induced attention over rows."""

    attn1: CrossAttentionBlock
    attn2: CrossAttentionBlock
    inducing_vectors: Array

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        num_inducing_points: int,
        mlp_ratio: float = 2.0,
        drop_path: float | list[float] = 0.0,
        scaling_mlp_hidden_dim: int = 64,
        act_layer: str | Callable = "exactgelu",
        norm_layer: str | type[eqx.Module] = "rmsnorm",
        eps: float = 1e-5,
    ) -> None:
        key_attn1, key_attn2, key_inducing = jr.split(key, 3)
        self.attn1 = CrossAttentionBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            use_softmax_scaling=True,
            scaling_mlp_hidden_dim=scaling_mlp_hidden_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            eps=eps,
            key=key_attn1,
        )
        self.attn2 = CrossAttentionBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            eps=eps,
            key=key_attn2,
        )
        self.inducing_vectors = (
            jr.truncated_normal(key_inducing, -2, 2, (num_inducing_points, dim)) * 0.02
        )

    def __call__(
        self,
        x: Float[Array, "rows dim"],
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
        n_train: int,
        **kwargs,
    ) -> Float[Array, "rows dim"]:
        key_attn1, key_attn2 = _split_optional_key(key, 2)
        hidden = self.attn1(
            self.inducing_vectors,
            x[:n_train],
            key=key_attn1,
            inference=inference,
        )
        return self.attn2(x, hidden, key=key_attn2, inference=inference)


@register_layer()
class FeatureDistributionEncoder(eqx.Module):
    """Column-wise stack of induced attention blocks."""

    blocks: BlockChunk
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        *,
        key: PRNGKeyArray,
        num_inducing_points: int = 128,
        mlp_ratio: float = 2.0,
        drop_path: float | Sequence[float] = 0.0,
        scaling_mlp_hidden_dim: int = 64,
        act_layer: str | Callable = "exactgelu",
        norm_layer: str | type[eqx.Module] = "rmsnorm",
        eps: float = 1e-5,
    ) -> None:
        self.blocks = BlockChunk(
            depth=depth,
            module=InducedAttentionBlock,
            module_kwargs={
                "dim": dim,
                "num_heads": num_heads,
                "num_inducing_points": num_inducing_points,
                "mlp_ratio": mlp_ratio,
                "scaling_mlp_hidden_dim": scaling_mlp_hidden_dim,
                "act_layer": act_layer,
                "norm_layer": norm_layer,
                "eps": eps,
            },
            drop_path=drop_path,
            key=key,
        )
        self.depth = depth

    def __call__(
        self,
        x: Float[Array, "rows columns dim"],
        n_train: int,
        *,
        key: PRNGKeyArray,
        inference: bool | None = None,
    ) -> Float[Array, "rows columns dim"]:
        x_columns = x.transpose(1, 0, 2)

        def encode_column(column: Float[Array, "rows dim"]) -> Float[Array, "rows dim"]:
            return self.blocks(
                column,
                n_train=n_train,
                inference=inference,
                key=key,
            )

        return jax.vmap(encode_column)(x_columns).transpose(1, 0, 2)


@register_layer()
class ColumnAggregator(eqx.Module):
    """Per-row cross-feature interaction with learned readout tokens."""

    blocks: BlockChunk | None
    readout_block: AttentionBlock
    cls_tokens: Array
    rope: eqx.nn.RotaryPositionalEmbedding | None
    norm: eqx.Module
    num_cls_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        *,
        key: PRNGKeyArray,
        num_cls_tokens: int = 4,
        mlp_ratio: float = 2.0,
        use_rope: bool = True,
        rope_base: float = 100_000.0,
        act_layer: str | Callable = "exactgelu",
        norm_layer: str | type[eqx.Module] = "rmsnorm",
        eps: float = 1e-5,
    ) -> None:
        if depth < 1:
            raise ValueError("ColumnAggregator requires depth >= 1.")

        key_blocks, key_readout, key_cls = jr.split(key, 3)
        self.blocks = (
            BlockChunk(
                depth=depth - 1,
                module=AttentionBlock,
                module_kwargs={
                    "dim": dim,
                    "num_heads": num_heads,
                    "mlp_ratio": mlp_ratio,
                    "act_layer": act_layer,
                    "norm_layer": norm_layer,
                    "eps": eps,
                },
                key=key_blocks,
            )
            if depth > 1
            else None
        )
        self.readout_block = AttentionBlock(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            eps=eps,
            key=key_readout,
        )
        self.cls_tokens = (
            jr.truncated_normal(key_cls, -2, 2, (num_cls_tokens, dim)) * 0.02
        )
        self.rope = (
            eqx.nn.RotaryPositionalEmbedding(dim // num_heads, theta=rope_base)
            if use_rope
            else None
        )
        self.norm = get_norm(norm_layer)(dim, eps=eps)
        self.num_cls_tokens = num_cls_tokens

    def _row(
        self,
        x: Float[Array, "columns dim"],
        *,
        key: PRNGKeyArray,
        inference: bool | None = None,
    ) -> Float[Array, "cls dim"]:
        key_blocks, key_readout = jr.split(key, 2)
        x = jnp.concatenate([self.cls_tokens, x], axis=0)
        if self.blocks is not None:
            x = self.blocks(
                x,
                rope=self.rope,
                inference=inference,
                key=key_blocks,
            )
        cls = self.readout_block.forward_cross(
            x[: self.num_cls_tokens],
            x,
            self.rope,
            key=key_readout,
            inference=inference,
        )
        return jax.vmap(self.norm)(cls)

    def __call__(
        self,
        x: Float[Array, "rows columns dim"],
        *,
        key: PRNGKeyArray,
        inference: bool | None = None,
    ) -> Float[Array, "rows cls dim"]:
        return jax.vmap(lambda row: self._row(row, key=key, inference=inference))(x)
