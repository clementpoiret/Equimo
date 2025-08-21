from typing import Callable, List, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.attention import (
    Attention,
    AttentionBlock,
    get_attention,
    get_attention_block,
)
from equimo.layers.ffn import Mlp, get_ffn
from equimo.layers.norm import get_norm
from equimo.layers.patch import PatchEmbedding
from equimo.layers.posemb import DinoRoPE, LearnedPosEmbed, PosCNN
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
            blocks.append(block(**config, key=block_subkeys[i]))
        self.blocks = blocks

        self.downsample = downsampler(dim=dim, **downsampler_kwargs, key=key_ds)

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "..."]:
        keys = jr.split(key, len(self.blocks))

        x = self.posemb(x)

        for blk, key_block in zip(self.blocks, keys):
            x = blk(x, inference=inference, key=key_block, **kwargs)

        if self.downsampler_contains_dropout:
            x = self.downsample(x, inference=inference, key=key)
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
    pos_embed: LearnedPosEmbed | DinoRoPE
    cls_token: jax.Array | None
    reg_tokens: jax.Array | None
    mask_token: jax.Array | None
    blocks: List[eqx.Module]
    pos_drop: eqx.nn.Dropout
    norm: eqx.Module
    local_cls_norm: eqx.Module | None
    head: eqx.Module

    dim: int = eqx.field(static=True)
    embed_size: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)
    global_pool: str = eqx.field(static=True)
    num_reg_tokens: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    num_embedded_prefix_tokens: int = eqx.field(static=True)
    no_embed_class: bool = eqx.field(static=True)
    pos_embed_reg_tokens: bool = eqx.field(static=True)
    use_rope_pos_embed: bool = eqx.field(static=True)
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
        use_mask_token: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 4,
        use_rope_pos_embed: bool = False,
        rope_pos_embed_base: float = 100.0,
        rope_pos_embed_min_period: Optional[float] = None,
        rope_pos_embed_max_period: Optional[float] = None,
        rope_pos_embed_normalize_coords: Literal["min", "max", "separate"] = "separate",
        rope_pos_embed_shift_coords: Optional[float] = None,
        rope_pos_embed_jitter_coords: Optional[float] = None,
        rope_pos_embed_rescale_coords: Optional[float] = None,
        rope_pos_embed_dtype: jnp.dtype = jnp.float32,
        pos_embed_reg_tokens: bool = False,
        pos_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        block: str | eqx.Module = AttentionBlock,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable | str = jax.nn.gelu,
        attn_layer: str | eqx.Module = Attention,
        ffn_layer: str | eqx.Module = Mlp,
        ffn_bias: bool = True,
        ffn_kwargs: dict = {},
        norm_layer: str | eqx.Module = eqx.nn.LayerNorm,
        untie_global_and_local_cls_norm: bool = False,
        init_values: float | None = None,
        global_pool: Literal["", "token", "avg", "avgmax", "max"] = "avg",
        num_classes: int = 1000,
        interpolate_antialias: bool = False,
        eps: float = 1e-5,
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
        self.embed_size = img_size // patch_size
        self.use_rope_pos_embed = use_rope_pos_embed

        block = get_attention_block(block)
        attn_layer = get_attention(attn_layer)
        ffn_layer = get_ffn(ffn_layer)
        norm_layer = get_norm(norm_layer)
        act_layer = get_act(act_layer)

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
        self.cls_token = jr.normal(key_cls, (1, dim)) if class_token else None
        self.reg_tokens = (
            jr.normal(key_reg, (reg_tokens, dim)) if reg_tokens > 0 else None
        )

        self.mask_token = jnp.zeros((1, dim)) if use_mask_token else None

        if no_embed_class:
            self.embed_len = self.num_patches
        elif self.pos_embed_reg_tokens:
            self.embed_len = self.num_patches + self.num_prefix_tokens
            self.num_embedded_prefix_tokens += self.num_prefix_tokens
        else:
            self.num_embedded_prefix_tokens += 1
            self.embed_len = self.num_patches + 1

        if use_rope_pos_embed:
            if not isinstance(num_heads, int):
                raise ValueError(
                    "RoPE pos embedding currently requires a static number of heads."
                )
            self.pos_embed = DinoRoPE(
                dim=dim,
                num_heads=num_heads,
                base=rope_pos_embed_base,
                min_period=rope_pos_embed_min_period,
                max_period=rope_pos_embed_max_period,
                normalize_coords=rope_pos_embed_normalize_coords,
                shift_coords=rope_pos_embed_shift_coords,
                jitter_coords=rope_pos_embed_jitter_coords,
                rescale_coords=rope_pos_embed_rescale_coords,
                dtype=rope_pos_embed_dtype,
            )
        else:
            self.pos_embed = LearnedPosEmbed(
                weight=jr.normal(key_posemb, (self.embed_len, dim)),
                dim=dim,
                embed_size=self.embed_size,
                num_prefix_tokens=self.num_prefix_tokens,
                num_embedded_prefix_tokens=self.num_embedded_prefix_tokens,
                no_embed_class=self.no_embed_class,
                pos_embed_reg_tokens=self.pos_embed_reg_tokens,
                antialias=interpolate_antialias,
            )
        self.pos_drop = eqx.nn.Dropout(pos_drop_rate)

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
                ffn_kwargs=ffn_kwargs,
                norm_layer=norm_layer,
                init_values=init_values,
                eps=eps,
                key=block_subkeys[i],
            )
            for i, depth in enumerate(depths)
        ]

        self.norm = norm_layer(dim, eps=eps)

        # WARNING: This has no effect in the code.
        # This norm layer is created to hold some training-only norm layer of Dinov3
        self.local_cls_norm = (
            norm_layer(dim, eps=eps) if untie_global_and_local_cls_norm else None
        )

        self.head = (
            eqx.nn.Linear(dim, num_classes, key=key_head)
            if num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        mask: Optional[Int[Array, "embed_h embed_w"]] = None,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        """Extract features from input image.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations
            mask: optional binary mask of the size of the input after patch embedding

        Returns:
            Processed feature tensor
        """
        key_pos, *block_subkeys = jr.split(key, len(self.blocks) + 1)
        x = self.patch_embed(x)

        if mask is not None:
            assert self.mask_token is not None, (
                "To use masked forward, init the model with `use_mask_token=True`."
            )
            if self.dynamic_img_size:
                mask = rearrange(mask, "h w -> 1 h w")
                value = rearrange(self.mask_token, "1 c -> c 1 1")
            else:
                mask = rearrange(mask, "h w -> (h w) 1")
                value = self.mask_token
            x = jnp.where(mask, x, value.astype(x.dtype))

        if self.use_rope_pos_embed:
            # In models like Dinov3, RoPE is not applied here, but in self attention blocks
            # It means that we have to dumbly cat prefix token and flattened x manually
            _, H, W = x.shape
            if inference:
                rope_sincos = self.pos_embed.get_sincos(
                    H=H, W=W, inference=inference, key=key_pos
                )
            x = jnp.concatenate(
                [
                    self.cls_token,
                    self.reg_tokens,
                    rearrange(x, "c h w -> (h w) c"),
                ],
                axis=0,
            )
        else:
            # TODO: pos drop
            rope_sincos = None
            x = self.pos_embed(
                x,
                cls_token=self.cls_token,
                reg_tokens=self.reg_tokens,
                dynamic_img_size=self.dynamic_img_size,
            )

        for blk, key_block in zip(self.blocks, block_subkeys):
            if self.use_rope_pos_embed and not inference:
                key_pos, key_rope = jr.split(key_pos, 2)
                rope_sincos = (
                    self.pos_embed.get_sincos(
                        H=H, W=W, inference=inference, key=key_rope
                    )
                    if self.use_rope_pos_embed
                    else None
                )
            x = blk(
                x, rope_sincos=rope_sincos, inference=inference, key=key_block, **kwargs
            )

        return x

    def forward_features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        """Process features and return intermediate representations.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Dictionary containing:
                - x_norm_cls_token: Normalized class token
                - x_norm_reg_tokens: Normalized registration tokens
                - x_norm_patchtokens: Normalized patch tokens
                - x_prenorm: Pre-normalized features
        """
        x = self.features(x, inference=inference, key=key, **kwargs)
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
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:
        """Process input image through the full network.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Classification logits
        """
        x = self.features(x, inference=inference, key=key, **kwargs)
        x = jax.vmap(self.norm)(x)
        x = pool_sd(
            x,
            num_prefix_tokens=self.num_prefix_tokens,
            pool_type=self.global_pool,
            reduce_include_prefix=False,
        )

        x = self.head(x)

        return x
