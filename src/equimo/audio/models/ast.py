# ty: ignore[invalid-assignment]
# ty: ignore[call-non-callable]
# ty: ignore[too-many-positional-arguments]
# ty: ignore[unknown-argument]
__all__ = [
    "AudioSpectrogramTransformer",
    "ast_tiny_patch16_224",
    "ast_small_patch16_224",
    "ast_base_patch16_224",
    "ast_base_patch16_384",
    "ast_base_patch16_audioset_10_10_0_4593",
    "ast_base_patch16_speechcommands_v2_10_10_0_9812",
]

from typing import Callable, Literal, Optional, Tuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.audio.layers.patch import SpectrogramPatchEmbedding
from equimo.core.layers.activation import get_act
from equimo.core.layers.attention import get_attn, get_attn_block
from equimo.core.layers.ffn import get_ffn
from equimo.core.layers.generic import BlockChunk
from equimo.core.layers.norm import get_norm
from equimo.registry import register_model
from equimo.utils import pool_sd, to_list


@register_model("ast", modality="audio")
class AudioSpectrogramTransformer(eqx.Module):
    """Audio Spectrogram Transformer (AST).

    Native Equinox implementation of the AST architecture for single
    spectrograms shaped ``(time, frequency)``. Batch inputs can be handled with
    ``jax.vmap``.
    """

    patch_embed: SpectrogramPatchEmbedding
    pos_embed: jax.Array
    cls_token: jax.Array
    dist_token: jax.Array
    blocks: Tuple[eqx.Module, ...]
    pos_drop: eqx.nn.Dropout
    norm: eqx.Module
    head_norm: eqx.Module
    head: eqx.Module

    dim: int = eqx.field(static=True)
    input_fdim: int = eqx.field(static=True)
    input_tdim: int = eqx.field(static=True)
    patch_size: Tuple[int, int] = eqx.field(static=True)
    fstride: int = eqx.field(static=True)
    tstride: int = eqx.field(static=True)
    grid_size: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    global_pool: str = eqx.field(static=True)

    def __init__(
        self,
        input_fdim: int,
        input_tdim: int,
        dim: int,
        patch_size: int | Tuple[int, int],
        fstride: int,
        tstride: int,
        num_heads: int | list[int],
        depths: list[int],
        *,
        key: PRNGKeyArray,
        pos_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        block: str | type[eqx.Module] = "attentionblock",
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: str | Callable = "gelu",
        attn_layer: str | type[eqx.Module] = "attention",
        ffn_layer: str | type[eqx.Module] = "mlp",
        ffn_bias: bool = True,
        ffn_kwargs: dict = {},
        norm_layer: str | type[eqx.Module] = "layernorm",
        head_norm_layer: str | type[eqx.Module] | None = None,
        init_values: float | None = None,
        global_pool: Literal[
            "token", "cls_patch_mean", "avg", "avgmax", "max"
        ] = "token",
        num_classes: int | None = 527,
        eps: float = 1e-5,
        **kwargs,
    ):
        depth = sum(depths)
        key_patch, key_pos, key_cls, key_dist, key_head, *block_subkeys = jr.split(
            key, 5 + len(depths)
        )

        self.dim = dim
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.fstride = fstride
        self.tstride = tstride
        self.num_prefix_tokens = 2
        self.global_pool = global_pool

        block = get_attn_block(block)
        attn_layer = get_attn(attn_layer)
        ffn_layer = get_ffn(ffn_layer)
        norm_layer = get_norm(norm_layer)
        act_layer = get_act(act_layer)

        self.patch_embed = SpectrogramPatchEmbedding(
            embed_dim=dim,
            patch_size=patch_size,
            input_fdim=input_fdim,
            input_tdim=input_tdim,
            fstride=fstride,
            tstride=tstride,
            key=key_patch,
        )
        self.patch_size = self.patch_embed.patch_size
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = jr.normal(key_cls, (1, dim))
        self.dist_token = jr.normal(key_dist, (1, dim))
        self.pos_embed = jr.normal(key_pos, (self.num_patches + 2, dim))
        self.pos_drop = eqx.nn.Dropout(pos_drop_rate)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        n_chunks = len(depths)
        num_heads = to_list(num_heads, n_chunks)
        attn_layer = to_list(attn_layer, n_chunks)
        self.blocks = tuple(
            BlockChunk(
                depth=depths[i],
                module=block,
                module_kwargs={
                    "dim": dim,
                    "num_heads": num_heads[i],
                    "mlp_ratio": mlp_ratio,
                    "qkv_bias": qkv_bias,
                    "proj_bias": proj_bias,
                    "qk_norm": qk_norm,
                    "attn_drop": attn_drop,
                    "proj_drop": proj_drop,
                    "act_layer": act_layer,
                    "attn_layer": attn_layer[i],
                    "ffn_layer": ffn_layer,
                    "ffn_bias": ffn_bias,
                    "ffn_kwargs": ffn_kwargs,
                    "norm_layer": norm_layer,
                    "eps": eps,
                },
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                init_values=init_values,
                key=block_subkeys[i],
            )
            for i in range(n_chunks)
        )

        self.norm = norm_layer(dim, eps=eps)
        head_in_features = 2 * dim if global_pool == "cls_patch_mean" else dim
        self.head_norm = (
            get_norm(head_norm_layer)(head_in_features, eps=eps)
            if head_norm_layer is not None
            else eqx.nn.Identity()
        )
        self.head = (
            eqx.nn.Linear(head_in_features, num_classes, key=key_head)
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "time frequency"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        key_pos, *block_subkeys = jr.split(key, len(self.blocks) + 1)

        x = self.patch_embed(x)
        prefix = [
            self.cls_token.astype(x.dtype),
            self.dist_token.astype(x.dtype),
        ]
        x = jnp.concatenate(prefix + [x], axis=0)
        x = x + self.pos_embed.astype(x.dtype)
        x = self.pos_drop(x, inference=inference, key=key_pos)

        for blk, key_block in zip(self.blocks, block_subkeys):
            x = blk(x, inference=inference, key=key_block, **kwargs)

        return x

    def forward_features(
        self,
        x: Float[Array, "time frequency"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        x = self.features(x, inference=inference, key=key, **kwargs)
        x_norm = jax.vmap(self.norm)(x)

        return {
            "x_norm_cls_token": x_norm[0],
            "x_norm_dist_token": x_norm[1],
            "x_norm_patchtokens": x_norm[self.num_prefix_tokens :],
            "x_prenorm": x,
        }

    def __call__(
        self,
        x: Float[Array, "time frequency"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:  # noqa: F821
        x = self.features(x, inference=inference, key=key, **kwargs)
        x = jax.vmap(self.norm)(x)

        if self.global_pool == "token":
            x = 0.5 * (x[0] + x[1])
        else:
            x = pool_sd(
                x,
                num_prefix_tokens=self.num_prefix_tokens,
                pool_type=self.global_pool,
                reduce_include_prefix=False,
            )

        x = self.head_norm(x)
        return self.head(x)


_AST_BASE_CFG: dict = {
    "input_fdim": 128,
    "input_tdim": 1024,
    "patch_size": 16,
    "fstride": 10,
    "tstride": 10,
    "num_classes": 527,
    "act_layer": "gelu",
}

_AST_REGISTRY: dict[str, tuple[dict, dict]] = {
    "ast_tiny_patch16_224": (
        _AST_BASE_CFG,
        {"dim": 192, "num_heads": [3], "depths": [12]},
    ),
    "ast_small_patch16_224": (
        _AST_BASE_CFG,
        {"dim": 384, "num_heads": [6], "depths": [12]},
    ),
    "ast_base_patch16_224": (
        _AST_BASE_CFG,
        {"dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "ast_base_patch16_384": (
        _AST_BASE_CFG,
        {"dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "ast_base_patch16_audioset_10_10_0_4593": (
        _AST_BASE_CFG,
        {
            "dim": 768,
            "num_heads": [12],
            "depths": [12],
            "act_layer": "exactgelu",
            "head_norm_layer": "layernorm",
            "eps": 1e-12,
        },
    ),
    "ast_base_patch16_speechcommands_v2_10_10_0_9812": (
        _AST_BASE_CFG,
        {
            "input_tdim": 128,
            "dim": 768,
            "num_heads": [12],
            "depths": [12],
            "num_classes": 35,
            "act_layer": "exactgelu",
            "head_norm_layer": "layernorm",
            "eps": 1e-12,
        },
    ),
}

_AST_PRETRAINED_VARIANTS = {
    "ast_base_patch16_audioset_10_10_0_4593",
    "ast_base_patch16_speechcommands_v2_10_10_0_9812",
}


def _build_ast(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> AudioSpectrogramTransformer:
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _AST_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    model = cast(
        AudioSpectrogramTransformer, AudioSpectrogramTransformer(**cfg, key=key)
    )

    if pretrained:
        if variant not in _AST_PRETRAINED_VARIANTS:
            supported = ", ".join(sorted(_AST_PRETRAINED_VARIANTS))
            raise ValueError(
                f"No pretrained weights are available for {variant!r}. "
                f"Supported AST pretrained variants: {supported}."
            )

        from equimo.serialization import load_weights

        return cast(
            AudioSpectrogramTransformer,
            load_weights(
                model,
                identifier=variant,
                inference_mode=inference_mode,
            ),
        )

    return model


def ast_tiny_patch16_224(**kwargs) -> AudioSpectrogramTransformer:
    """AST-Ti/16 - 192-dim, 3 heads, 12 blocks, patch 16."""
    return _build_ast("ast_tiny_patch16_224", **kwargs)


def ast_small_patch16_224(**kwargs) -> AudioSpectrogramTransformer:
    """AST-S/16 - 384-dim, 6 heads, 12 blocks, patch 16."""
    return _build_ast("ast_small_patch16_224", **kwargs)


def ast_base_patch16_224(**kwargs) -> AudioSpectrogramTransformer:
    """AST-B/16 - 768-dim, 12 heads, 12 blocks, patch 16."""
    return _build_ast("ast_base_patch16_224", **kwargs)


def ast_base_patch16_384(**kwargs) -> AudioSpectrogramTransformer:
    """AST-B/16 - base architecture matching the reference ``base384`` size."""
    return _build_ast("ast_base_patch16_384", **kwargs)


def ast_base_patch16_audioset_10_10_0_4593(
    **kwargs,
) -> AudioSpectrogramTransformer:
    """AST-B/16 fine-tuned on AudioSet with 10x10 strides (0.4593 mAP)."""
    return _build_ast("ast_base_patch16_audioset_10_10_0_4593", **kwargs)


def ast_base_patch16_speechcommands_v2_10_10_0_9812(
    **kwargs,
) -> AudioSpectrogramTransformer:
    """AST-B/16 fine-tuned on SpeechCommands V2 with 10x10 strides (98.12%)."""
    return _build_ast("ast_base_patch16_speechcommands_v2_10_10_0_9812", **kwargs)
