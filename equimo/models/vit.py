import math
from typing import Callable, List, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.attention import Attention, AttentionBlock
from equimo.layers.dropout import Dropout
from equimo.layers.ffn import Mlp
from equimo.layers.patch import PatchEmbedding
from equimo.layers.posemb import PosCNN
from equimo.layers.sharing import LayerSharing
from equimo.utils import pool_sd, to_list


class BlockChunk(eqx.Module):
    """A chunk of transformer blocks with optional downsampling.

    Processes input features through a sequence of transformer blocks with shared
    parameters, optionally applying positional embeddings and downsampling.

    Attributes:
        reshape: Whether to reshape inputs for processing
        downsampler_contains_dropout: If downsampler has dropout
        posemb: Positional embedding layer
        blocks: List of processing blocks
        downsample: Downsampling layer
    """

    reshape: bool = eqx.field(static=True)
    downsampler_contains_dropout: bool = eqx.field(static=True)

    posemb: eqx.Module
    blocks: List[eqx.Module]
    downsample: eqx.Module

    def __init__(
        self,
        depth: int,
        *,
        key: PRNGKeyArray,
        block: eqx.Module = AttentionBlock,
        repeat: int = 1,
        use_cpe: bool = False,
        downsampler: eqx.Module = eqx.nn.Identity,
        downsampler_contains_dropout: bool = False,
        downsampler_kwargs: dict = {},
        **kwargs,
    ):
        key_ds, key_pos, *block_subkeys = jr.split(key, depth + 2)
        if not isinstance(downsampler, eqx.nn.Identity) or use_cpe:
            if kwargs.get("dim") is None:
                raise ValueError(
                    "Using a downsampler or a CPE requires passing a `dim` argument."
                )

        # self.reshape = block is not ConvBlock
        self.reshape = True  # TODO
        self.downsampler_contains_dropout = downsampler_contains_dropout

        keys_to_spread = [
            k for k, v in kwargs.items() if isinstance(v, list) and len(v) == depth
        ]

        dim = kwargs.get("dim")
        self.posemb = (
            PosCNN(
                dim,
                dim,
                key=key_pos,
            )
            if use_cpe
            else eqx.nn.Identity()
        )

        blocks = []
        for i in range(depth):
            config = kwargs | {k: kwargs[k][i] for k in keys_to_spread}
            blocks.append(
                LayerSharing(
                    dim=dim,
                    f=block(**config, key=block_subkeys[i]),
                    repeat=repeat,
                    key=block_subkeys[i],
                ),
            )
        self.blocks = blocks

        self.downsample = downsampler(dim=dim, **downsampler_kwargs, key=key_ds)

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        enable_dropout: bool,
        key: PRNGKeyArray,
        **kwargs,
    ) -> Float[Array, "..."]:
        keys = jr.split(key, len(self.blocks))

        x = self.posemb(x)

        for blk, key_block in zip(self.blocks, keys):
            x = blk(x, enable_dropout=enable_dropout, key=key_block, **kwargs)

        if self.downsampler_contains_dropout:
            x = self.downsample(x, enable_dropout, key)
        else:
            x = self.downsample(x)

        return x


class VisionTransformer(eqx.Module):
    """Vision Transformer (ViT) implementation.

    A transformer architecture for image processing that divides input images into patches,
    processes them through transformer blocks, and includes options for class tokens,
    registration tokens, and various pooling strategies.

    Attributes:
        patch_embed: Patch embedding layer
        pos_embed: Positional embedding array
        cls_token: Class token for classification (optional)
        reg_tokens: Registration tokens for alignment (optional)
        blocks: List of transformer blocks
        pos_drop: Positional dropout layer
        norm: Normalization layer
        head: Classification head
        dim: Model dimension
        num_patches: Number of image patches
        global_pool: Global pooling strategy
        num_reg_tokens: Number of registration tokens
        num_prefix_tokens: Total number of prefix tokens
        num_embedded_prefix_tokens: Number of embedded prefix tokens
        no_embed_class: Whether to skip class token embedding
        pos_embed_reg_tokens: Whether to add positional embeddings to reg tokens
        embed_len: Total embedding length
        dynamic_img_size: Whether to support dynamic image sizes
        antialias: Whether to use antialiasing in interpolation
    """

    patch_embed: PatchEmbedding
    pos_embed: jnp.ndarray
    cls_token: jnp.ndarray | None
    reg_tokens: jnp.ndarray | None
    blocks: List[eqx.Module]
    pos_drop: Dropout
    norm: eqx.Module
    head: eqx.Module

    dim: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    global_pool: str = eqx.field(static=True)
    num_reg_tokens: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    num_embedded_prefix_tokens: int = eqx.field(static=True)
    no_embed_class: bool = eqx.field(static=True)
    pos_embed_reg_tokens: bool = eqx.field(static=True)
    embed_len: int = eqx.field(static=True)
    dynamic_img_size: bool = eqx.field(static=True)
    antialias: bool = eqx.field(static=True)

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        dim: int,
        patch_size: int,
        num_heads: int | List[int],
        depths: List[int],
        *,
        key: PRNGKeyArray,
        repeat: int = 1,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 4,
        pos_embed_reg_tokens: bool = False,
        pos_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        block: eqx.Module = AttentionBlock,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = jax.nn.gelu,
        attn_layer: eqx.Module = Attention,
        ffn_layer: eqx.Module = Mlp,
        ffn_bias: bool = True,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        init_values: float | None = None,
        global_pool: Literal["", "token", "avg", "avgmax", "max"] = "avg",
        num_classes: int = 1000,
        interpolate_antialias: bool = False,
        **kwargs,
    ):
        depth = sum(depths)
        key_patchemb, key_posemb, key_cls, key_reg, key_head, *block_subkeys = jr.split(
            key, 5 + len(depths)
        )
        self.dim = dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.num_embedded_prefix_tokens = 0
        self.dynamic_img_size = dynamic_img_size
        self.antialias = interpolate_antialias
        self.no_embed_class = no_embed_class
        self.pos_embed_reg_tokens = pos_embed_reg_tokens
        self.global_pool = global_pool

        self.patch_embed = PatchEmbedding(
            in_channels,
            dim,
            patch_size,
            img_size=img_size,
            flatten=not dynamic_img_size,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            key=key_patchemb,
        )
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = jr.normal(key_cls, (1, dim))
        self.reg_tokens = jr.normal(key_reg, (reg_tokens, dim))

        if no_embed_class:
            self.embed_len = self.num_patches
        elif self.pos_embed_reg_tokens:
            self.embed_len = self.num_patches + self.num_prefix_tokens
            self.num_embedded_prefix_tokens += self.num_prefix_tokens
        else:
            self.num_embedded_prefix_tokens += 1
            self.embed_len = self.num_patches + 1

        self.pos_embed = jr.normal(key_posemb, (self.embed_len, dim))
        self.pos_drop = Dropout(pos_drop_rate)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = list(jnp.linspace(0.0, drop_path_rate, depth))

        n_chunks = len(depths)
        dims = to_list(dim, n_chunks)
        num_heads = to_list(num_heads, n_chunks)
        attn_layer = to_list(attn_layer, n_chunks)
        self.blocks = [
            BlockChunk(
                block=block,
                repeat=repeat,
                dim=dims[i],
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                act_layer=act_layer,
                attn_layer=attn_layer[i],
                ffn_layer=ffn_layer,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                init_values=init_values,
                key=block_subkeys[i],
            )
            for i, depth in enumerate(depths)
        ]

        self.norm = norm_layer(dim)
        self.head = (
            eqx.nn.Linear(dim, num_classes, key=key_head)
            if num_classes > 0
            else eqx.nn.Identity()
        )

    def resample_pos_embed(
        self,
        pos_embed: Float[Array, "embed_len dim"],
        new_size: Tuple[int, int],
        old_size: Optional[Tuple[int, int]] = None,
        interpolation: str = "bicubic",
        antialias: bool = True,
    ):
        """Resample positional embeddings for different input sizes.

        Args:
            pos_embed: Original positional embeddings
            new_size: Target size (height, width)
            old_size: Original size (height, width), computed if None
            interpolation: Interpolation method
            antialias: Whether to use antialiasing

        Returns:
            Resampled positional embeddings
        """
        previous_dtype = pos_embed.dtype

        num_new_tokens = new_size[0] * new_size[1] + self.num_embedded_prefix_tokens

        if num_new_tokens == self.embed_len and new_size[0] == new_size[1]:
            return pos_embed

        if old_size is None:
            hw = int(math.sqrt(self.num_patches))
            old_size = hw, hw

        prefix_embed = (
            pos_embed[: self.num_prefix_tokens] if self.num_prefix_tokens else None
        )
        pos_embed = pos_embed[self.num_prefix_tokens :]

        pos_embed = pos_embed.astype("float32")
        pos_embed = rearrange(
            pos_embed, "(h w) d -> h w d", h=old_size[0], w=old_size[1]
        )
        pos_embed = jax.image.resize(
            pos_embed,
            (new_size[0], new_size[1], self.dim),
            method=interpolation,
            antialias=antialias,
        )
        pos_embed = rearrange(pos_embed, "h w d -> (h w) d").astype(previous_dtype)

        if prefix_embed is not None:
            pos_embed = jnp.concatenate([prefix_embed, pos_embed], axis=0)

        return pos_embed

    def _pos_embed(self, x: Float[Array, "..."], h: int, w: int):
        """Add positional embeddings to input features.

        Args:
            x: Input features
            h: Height of feature map
            w: Width of feature map

        Returns:
            Features with positional embeddings and tokens added
        """
        if self.pos_embed is None:
            return rearrange(x, "c h w -> (h w) c")

        if self.dynamic_img_size:
            C, H, W = x.shape
            pos_embed = self.resample_pos_embed(
                self.pos_embed, new_size=(H, W), antialias=self.antialias
            )
            x = rearrange("c h w -> (h w) c")
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token)
        if self.reg_tokens is not None:
            to_cat.append(self.reg_tokens)

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=0)
        elif self.pos_embed_reg_tokens:
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=0)
            x = x + pos_embed
        else:
            x = jnp.concatenate(to_cat[:1] + [x], axis=0)  # cat cls_token
            x = x + pos_embed
            if self.reg_tokens is not None:
                x = jnp.concatenate(
                    [x[:1], to_cat[1], x[1:]], axis=0
                )  # insert reg_tokens in between

        return x

    def features(
        self,
        x: Float[Array, "channels height width"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "seqlen dim"]:
        """Extract features from input image.

        Args:
            x: Input image tensor
            enable_dropout: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Processed feature tensor
        """
        key_posdrop, *block_subkeys = jr.split(key, len(self.blocks) + 1)
        x = self.patch_embed(x)
        x = self.pos_drop(x, inference=not enable_dropout, key=key_posdrop)

        for blk, key_block in zip(self.blocks, block_subkeys):
            x = blk(x, enable_dropout=enable_dropout, key=key_block)

        return x

    def forward_features(
        self,
        x: Float[Array, "channels height width"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> dict:
        """Process features and return intermediate representations.

        Args:
            x: Input image tensor
            enable_dropout: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Dictionary containing:
                - x_norm_cls_token: Normalized class token
                - x_norm_reg_tokens: Normalized registration tokens
                - x_norm_patchtokens: Normalized patch tokens
                - x_prenorm: Pre-normalized features
        """
        key_posdrop, *block_subkeys = jr.split(key, len(self.blocks) + 1)
        x = self.patch_embed(x)
        x = self.pos_drop(x, inference=not enable_dropout, key=key_posdrop)

        for blk, key_block in zip(self.blocks, block_subkeys):
            x = blk(x, enable_dropout=enable_dropout, key=key_block)

        x_norm = jax.vmap(self.norm)(x)

        return {
            "x_norm_cls_token": x_norm[0],
            "x_norm_reg_tokens": x_norm[1 : self.num_reg_tokens + 1],
            "x_norm_patchtokens": x_norm[self.num_reg_tokens + 1 :],
            "x_prenorm": x,
        }

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        enable_dropout: bool = False,
        key: PRNGKeyArray = jr.PRNGKey(42),
    ) -> Float[Array, "num_classes"]:
        """Process input image through the full network.

        Args:
            x: Input image tensor
            enable_dropout: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Classification logits
        """
        x = self.features(x, enable_dropout, key)
        x = jax.vmap(self.norm)(x)
        x = pool_sd(
            x,
            num_prefix_tokens=self.num_prefix_tokens,
            pool_type=self.global_pool,
            reduce_include_prefix=False,
        )

        x = self.head(x)

        return x
