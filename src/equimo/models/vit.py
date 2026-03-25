__all__ = [
    "VisionTransformer",
    # Standard ViT presets
    "vit_tiny_patch16_224",
    "vit_tiny_patch32_224",
    "vit_small_patch16_224",
    "vit_small_patch32_224",
    "vit_base_patch16_224",
    "vit_base_patch32_224",
    "vit_large_patch16_224",
    "vit_large_patch32_224",
    "vit_huge_patch14_224",
    "vit_huge_patch16_224",
    # DINOv2
    "dinov2_vits14",
    "dinov2_vits14_reg",
    "dinov2_vitb14",
    "dinov2_vitb14_reg",
    "dinov2_vitl14",
    "dinov2_vitl14_reg",
    "dinov2_vitg14",
    "dinov2_vitg14_reg",
    # DINOv3
    "dinov3_vits16_pretrain_lvd1689m",
    "dinov3_vits16plus_pretrain_lvd1689m",
    "dinov3_vitb16_pretrain_lvd1689m",
    "dinov3_vitl16_pretrain_lvd1689m",
    "dinov3_vith16plus_pretrain_lvd1689m",
    "dinov3_vit7b16_pretrain_lvd1689m",
    "dinov3_vitl16_pretrain_sat493m",
    "dinov3_vit7b16_pretrain_sat493m",
    # EUPE
    "eupe_vitt16",
    "eupe_vits16",
    "eupe_vitb16",
    # SigLIP2
    "siglip2_vitb16_224",
    "siglip2_vitb16_256",
    "siglip2_vitb16_384",
    "siglip2_vitb16_512",
    "siglip2_vitb32_256",
    "siglip2_vitl16_256",
    "siglip2_vitl16_384",
    "siglip2_vitl16_512",
    "siglip2_vitso400m14_224",
    "siglip2_vitso400m14_378",
    "siglip2_vitso400m16_256",
    "siglip2_vitso400m16_384",
    "siglip2_vitso400m16_512",
    "siglip2_vitgiantopt16_256",
    "siglip2_vitgiantopt16_384",
    # TIPS
    "tips_vits14_hr",
    "tips_vitb14_hr",
    "tips_vitl14_hr",
    "tips_vitso400m14_hr",
    "tips_vitg14_lr",
    "tips_vitg14_hr",
]

from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.attention import (
    get_attn,
    get_attn_block,
)
from equimo.layers.ffn import get_ffn
from equimo.layers.generic import BlockChunk
from equimo.layers.norm import get_norm
from equimo.layers.patch import PatchEmbedding
from equimo.layers.posemb import LearnedPosEmbed, VisionRoPE
from equimo.models.registry import register_model
from equimo.utils import pool_sd, to_list


@register_model("vit")
class VisionTransformer(eqx.Module):
    """Vision Transformer (ViT) implementation.

    A transformer architecture for image processing that divides input images into patches,
    processes them through transformer blocks, and includes options for class tokens,
    registration tokens, and various pooling strategies.

    Attributes:
        patch_embed: Patch embedding layer
        global_pos_embed: Model-level positional embedding applied after patching (e.g. APE)
        local_pos_embed: Block-level positional embedding passed to each attention block (e.g. RoPE)
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
    global_pos_embed: LearnedPosEmbed | None
    local_pos_embed: VisionRoPE | None
    cls_token: jax.Array | None
    reg_tokens: jax.Array | None
    mask_token: jax.Array | None
    blocks: Tuple[eqx.Module, ...]
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
    embed_len: int = eqx.field(static=True)
    dynamic_img_size: bool = eqx.field(static=True)
    antialias: bool = eqx.field(static=True)

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        dim: int,
        patch_size: int,
        num_heads: int | list[int],
        depths: list[int],
        *,
        key: PRNGKeyArray,
        use_mask_token: bool = False,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 4,
        use_global_pos_embed: bool = True,
        use_local_pos_embed: bool = False,
        local_pos_embed_config: dict = {
            "strategy": "period",
            "base": 100.0,
            "normalize_coords": "separate",
            "dtype": jnp.float32,
        },
        pos_embed_reg_tokens: bool = False,
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
        untie_global_and_local_cls_norm: bool = False,
        init_values: float | None = None,
        global_pool: Literal["", "token", "avg", "avgmax", "max"] = "avg",
        num_classes: int | None = 1000,
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

        block = get_attn_block(block)
        attn_layer = get_attn(attn_layer)
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

        if use_global_pos_embed:
            self.global_pos_embed = LearnedPosEmbed(
                weight=jr.normal(key_posemb, (self.embed_len, dim)),
                dim=dim,
                embed_size=self.embed_size,
                num_prefix_tokens=self.num_prefix_tokens,
                num_embedded_prefix_tokens=self.num_embedded_prefix_tokens,
                no_embed_class=self.no_embed_class,
                pos_embed_reg_tokens=self.pos_embed_reg_tokens,
                antialias=interpolate_antialias,
            )
        else:
            self.global_pos_embed = None

        if use_local_pos_embed:
            if not isinstance(num_heads, int):
                raise ValueError(
                    "Local pos embedding (RoPE) currently requires a static number of heads."
                )
            self.local_pos_embed = VisionRoPE(
                dim=dim,
                num_heads=num_heads,
                **local_pos_embed_config,
            )
        else:
            self.local_pos_embed = None
        self.pos_drop = eqx.nn.Dropout(pos_drop_rate)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        n_chunks = len(depths)
        dims = to_list(dim, n_chunks)
        num_heads = to_list(num_heads, n_chunks)
        attn_layer = to_list(attn_layer, n_chunks)
        self.blocks = tuple(
            BlockChunk(
                depth=depths[i],
                module=block,
                module_kwargs={
                    "dim": dims[i],
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
            for i, depth in enumerate(depths)
        )

        self.norm = norm_layer(dim, eps=eps)

        # WARNING: This has no effect in the code.
        # This norm layer is created to hold some training-only norm layer of Dinov3
        self.local_cls_norm = (
            norm_layer(dim, eps=eps) if untie_global_and_local_cls_norm else None
        )

        self.head = (
            eqx.nn.Linear(dim, num_classes, key=key_head)
            if num_classes is not None and num_classes > 0
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

        # Resolve spatial dims for local pos embed before flattening
        if self.local_pos_embed is not None:
            if self.dynamic_img_size:
                _, H, W = x.shape
            else:
                H = W = self.embed_size

        # Apply global (model-level) positional embedding (e.g. APE)
        if self.global_pos_embed is not None:
            x = self.global_pos_embed(
                x,
                cls_token=self.cls_token,
                reg_tokens=self.reg_tokens,
                dynamic_img_size=self.dynamic_img_size,
            )
        else:
            # No global pos embed: manually cat prefix tokens and flatten
            prefix = [t for t in (self.cls_token, self.reg_tokens) if t is not None]
            if self.dynamic_img_size:
                x = rearrange(x, "c h w -> (h w) c")
            x = jnp.concatenate([*prefix, x], axis=0) if prefix else x

        # Compute local (block-level) positional embedding (e.g. RoPE)
        rope_sincos = None
        if self.local_pos_embed is not None and inference:
            rope_sincos = self.local_pos_embed.get_sincos(
                H=H, W=W, inference=inference, key=key_pos
            )

        for blk, key_block in zip(self.blocks, block_subkeys):
            if self.local_pos_embed is not None and not inference:
                key_pos, key_rope = jr.split(key_pos, 2)
                rope_sincos = self.local_pos_embed.get_sincos(
                    H=H, W=W, inference=inference, key=key_rope
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


_VIT_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "num_classes": 1000,
    "reg_tokens": 0,
    "use_mask_token": False,
    "dynamic_img_size": False,
    "act_layer": "gelu",
}
_DINOV2_BASE_CFG: dict = {
    "img_size": 518,
    "in_channels": 3,
    "patch_size": 14,
    "num_classes": 0,
    "use_mask_token": True,
    "init_values": 1e-5,
    "eps": 1e-6,
    "dynamic_img_size": False,
    "act_layer": "exactgelu",
}
_DINOV3_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 0,
    "use_mask_token": True,
    "use_global_pos_embed": False,
    "use_local_pos_embed": True,
    "reg_tokens": 4,
    "init_values": 1e-5,
    "eps": 1e-5,
    "dynamic_img_size": True,
    "act_layer": "exactgelu",
}
_EUPE_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 0,
    "use_mask_token": True,
    "use_global_pos_embed": False,
    "use_local_pos_embed": True,
    "local_pos_embed_config": {
        "strategy": "period",
        "base": 100.0,
        "normalize_coords": "separate",
        "rescale_coords": 2.0,
    },
    "reg_tokens": 4,
    "init_values": 1e-5,
    "eps": 1e-5,
    "dynamic_img_size": True,
    "act_layer": "exactgelu",
}
_SIGLIP2_BASE_CFG: dict = {
    "img_size": 384,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 0,
    "use_mask_token": False,
    "reg_tokens": 0,
    "class_token": False,
    "no_embed_class": True,
    "init_values": None,
    "eps": 1e-6,
    "dynamic_img_size": False,
    "act_layer": "gelu",
}
_TIPS_BASE_CFG: dict = {
    "in_channels": 3,
    "patch_size": 14,
    "num_classes": 0,
    "use_mask_token": True,
    "reg_tokens": 1,
    "init_values": 1e-5,
    "eps": 1e-6,
    "dynamic_img_size": False,
    "act_layer": "exactgelu",
}
_VIT5_BASE_CFG: dict = {
    "img_size": 224,
    "in_channels": 3,
    "patch_size": 16,
    "num_classes": 1000,
    "reg_tokens": 4,
    "use_mask_token": False,
    "dynamic_img_size": False,
    "act_layer": "gelu",
    "use_global_pos_embed": True,
    "use_local_pos_embed": True,
    "init_values": 1e-4,
    "norm_layer": "rmsnorm",
    "qkv_bias": False,
    "qk_norm": True,
}

_VIT_REGISTRY: dict[str, tuple[dict, dict]] = {
    # Standard ViT (Dosovitskiy et al. + DeiT-III Ti/S)
    "vit_tiny_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 192, "patch_size": 16, "num_heads": [3], "depths": [12]},
    ),
    "vit_tiny_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 192, "patch_size": 32, "num_heads": [3], "depths": [12]},
    ),
    "vit_small_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 384, "patch_size": 16, "num_heads": [6], "depths": [12]},
    ),
    "vit_small_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 384, "patch_size": 32, "num_heads": [6], "depths": [12]},
    ),
    "vit_base_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 768, "patch_size": 16, "num_heads": [12], "depths": [12]},
    ),
    "vit_base_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 768, "patch_size": 32, "num_heads": [12], "depths": [12]},
    ),
    "vit_large_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 1024, "patch_size": 16, "num_heads": [16], "depths": [24]},
    ),
    "vit_large_patch32_224": (
        _VIT_BASE_CFG,
        {"dim": 1024, "patch_size": 32, "num_heads": [16], "depths": [24]},
    ),
    "vit_huge_patch14_224": (
        _VIT_BASE_CFG,
        {"dim": 1280, "patch_size": 14, "num_heads": [16], "depths": [32]},
    ),
    "vit_huge_patch16_224": (
        _VIT_BASE_CFG,
        {"dim": 1280, "patch_size": 16, "num_heads": [16], "depths": [32]},
    ),
    # DINOv2
    "dinov2_vits14": (
        _DINOV2_BASE_CFG,
        {"dim": 384, "num_heads": [6], "depths": [12], "reg_tokens": 0},
    ),
    "dinov2_vits14_reg": (
        _DINOV2_BASE_CFG,
        {"dim": 384, "num_heads": [6], "depths": [12], "reg_tokens": 4},
    ),
    "dinov2_vitb14": (
        _DINOV2_BASE_CFG,
        {"dim": 768, "num_heads": [12], "depths": [12], "reg_tokens": 0},
    ),
    "dinov2_vitb14_reg": (
        _DINOV2_BASE_CFG,
        {"dim": 768, "num_heads": [12], "depths": [12], "reg_tokens": 4},
    ),
    "dinov2_vitl14": (
        _DINOV2_BASE_CFG,
        {"dim": 1024, "num_heads": [16], "depths": [24], "reg_tokens": 0},
    ),
    "dinov2_vitl14_reg": (
        _DINOV2_BASE_CFG,
        {"dim": 1024, "num_heads": [16], "depths": [24], "reg_tokens": 4},
    ),
    "dinov2_vitg14": (
        _DINOV2_BASE_CFG,
        {
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "reg_tokens": 0,
            "ffn_layer": "swiglufused",
        },
    ),
    "dinov2_vitg14_reg": (
        _DINOV2_BASE_CFG,
        {
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "reg_tokens": 4,
            "ffn_layer": "swiglufused",
        },
    ),
    # DINOv3 (LVD-1689M)
    "dinov3_vits16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {"dim": 384, "num_heads": 6, "depths": [12]},
    ),
    "dinov3_vits16plus_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 384,
            "num_heads": 6,
            "depths": [12],
            "mlp_ratio": 6.0,
            "ffn_layer": "swiglu",
        },
    ),
    "dinov3_vitb16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {"dim": 768, "num_heads": 12, "depths": [12]},
    ),
    "dinov3_vitl16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {"dim": 1024, "num_heads": 16, "depths": [24]},
    ),
    "dinov3_vith16plus_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 1280,
            "num_heads": 20,
            "depths": [32],
            "mlp_ratio": 6.0,
            "ffn_layer": "swiglu",
        },
    ),
    "dinov3_vit7b16_pretrain_lvd1689m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 4096,
            "num_heads": 32,
            "depths": [40],
            "mlp_ratio": 3.0,
            "untie_global_and_local_cls_norm": True,
            "ffn_layer": "swiglu",
            "ffn_kwargs": {"align_to": 64},
        },
    ),
    # DINOv3 (SAT-493M)
    "dinov3_vitl16_pretrain_sat493m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 1024,
            "num_heads": 16,
            "depths": [24],
            "untie_global_and_local_cls_norm": True,
        },
    ),
    "dinov3_vit7b16_pretrain_sat493m": (
        _DINOV3_BASE_CFG,
        {
            "dim": 4096,
            "num_heads": 32,
            "depths": [40],
            "mlp_ratio": 3.0,
            "untie_global_and_local_cls_norm": True,
            "ffn_layer": "swiglu",
            "ffn_kwargs": {"align_to": 64},
        },
    ),
    # EUPE
    "eupe_vitt16": (
        _EUPE_BASE_CFG,
        {"dim": 192, "num_heads": 3, "depths": [12]},
    ),
    "eupe_vits16": (
        _EUPE_BASE_CFG,
        {"dim": 384, "num_heads": 6, "depths": [12]},
    ),
    "eupe_vitb16": (
        _EUPE_BASE_CFG,
        {"dim": 768, "num_heads": 12, "depths": [12]},
    ),
    # SigLIP2
    "siglip2_vitb16_224": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 224, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb16_256": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 256, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb16_384": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 384, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb16_512": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 512, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "siglip2_vitb32_256": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 256,
            "patch_size": 32,
            "dim": 768,
            "num_heads": [12],
            "depths": [12],
        },
    ),
    "siglip2_vitl16_256": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 256, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "siglip2_vitl16_384": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 384, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "siglip2_vitl16_512": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 512, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "siglip2_vitso400m14_224": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 224,
            "patch_size": 14,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m14_378": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 378,
            "patch_size": 14,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m16_256": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 256,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m16_384": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 384,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitso400m16_512": (
        _SIGLIP2_BASE_CFG,
        {
            "img_size": 512,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "siglip2_vitgiantopt16_256": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 256, "dim": 1536, "num_heads": [16], "depths": [40]},
    ),
    "siglip2_vitgiantopt16_384": (
        _SIGLIP2_BASE_CFG,
        {"img_size": 384, "dim": 1536, "num_heads": [16], "depths": [40]},
    ),
    # TIPS
    "tips_vits14_hr": (
        _TIPS_BASE_CFG,
        {"img_size": 448, "dim": 384, "num_heads": [6], "depths": [12]},
    ),
    "tips_vitb14_hr": (
        _TIPS_BASE_CFG,
        {"img_size": 448, "dim": 768, "num_heads": [12], "depths": [12]},
    ),
    "tips_vitl14_hr": (
        _TIPS_BASE_CFG,
        {"img_size": 448, "dim": 1024, "num_heads": [16], "depths": [24]},
    ),
    "tips_vitso400m14_hr": (
        _TIPS_BASE_CFG,
        {
            "img_size": 448,
            "dim": 1152,
            "num_heads": [16],
            "depths": [27],
            "mlp_ratio": 4304 / 1152,
        },
    ),
    "tips_vitg14_lr": (
        _TIPS_BASE_CFG,
        {
            "img_size": 224,
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "ffn_layer": "swiglufused",
        },
    ),
    "tips_vitg14_hr": (
        _TIPS_BASE_CFG,
        {
            "img_size": 448,
            "dim": 1536,
            "num_heads": [24],
            "depths": [40],
            "ffn_layer": "swiglufused",
        },
    ),
}


def _build_vit(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> VisionTransformer:
    """Construct a :class:`VisionTransformer` from the unified registry and
    optionally load pretrained weights.

    Args:
        variant: A key in :data:`_VIT_REGISTRY`.
        pretrained: If ``True``, download and deserialise the pretrained
            checkpoint from the default repository.
        inference_mode: Passed to :func:`equimo.io.load_weights` when
            *pretrained* is ``True``.  Defaults to ``True``.
        key: PRNG key for parameter initialisation.  Defaults to
            ``PRNGKey(42)`` when ``None``.
        **overrides: Extra keyword arguments merged into the model config,
            overriding stored defaults (e.g. ``num_classes=10``).

    Returns:
        A :class:`VisionTransformer` instance.

    Raises:
        KeyError: If *variant* is not found in the registry.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _VIT_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    model = VisionTransformer(**cfg, key=key)

    if pretrained:
        from equimo.io import load_weights

        model = load_weights(
            model,
            identifier=variant,
            inference_mode=inference_mode,
        )

    return model


def vit_tiny_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-Ti/16 — 192-dim, 3 heads, 12 blocks, patch 16, 224*224."""
    return _build_vit("vit_tiny_patch16_224", **kwargs)


def vit_tiny_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-Ti/32 — 192-dim, 3 heads, 12 blocks, patch 32, 224*224."""
    return _build_vit("vit_tiny_patch32_224", **kwargs)


def vit_small_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-S/16 — 384-dim, 6 heads, 12 blocks, patch 16, 224*224."""
    return _build_vit("vit_small_patch16_224", **kwargs)


def vit_small_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-S/32 — 384-dim, 6 heads, 12 blocks, patch 32, 224*224."""
    return _build_vit("vit_small_patch32_224", **kwargs)


def vit_base_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-B/16 — 768-dim, 12 heads, 12 blocks, patch 16, 224*224."""
    return _build_vit("vit_base_patch16_224", **kwargs)


def vit_base_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-B/32 — 768-dim, 12 heads, 12 blocks, patch 32, 224*224."""
    return _build_vit("vit_base_patch32_224", **kwargs)


def vit_large_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-L/16 — 1024-dim, 16 heads, 24 blocks, patch 16, 224*224."""
    return _build_vit("vit_large_patch16_224", **kwargs)


def vit_large_patch32_224(**kwargs) -> VisionTransformer:
    """ViT-L/32 — 1024-dim, 16 heads, 24 blocks, patch 32, 224*224."""
    return _build_vit("vit_large_patch32_224", **kwargs)


def vit_huge_patch14_224(**kwargs) -> VisionTransformer:
    """ViT-H/14 — 1280-dim, 16 heads, 32 blocks, patch 14, 224*224."""
    return _build_vit("vit_huge_patch14_224", **kwargs)


def vit_huge_patch16_224(**kwargs) -> VisionTransformer:
    """ViT-H/16 — 1280-dim, 16 heads, 32 blocks, patch 16, 224*224."""
    return _build_vit("vit_huge_patch16_224", **kwargs)


def dinov2_vits14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-S/14 (no register tokens)."""
    return _build_vit("dinov2_vits14", pretrained=pretrained, **kwargs)


def dinov2_vits14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-S/14 with 4 register tokens."""
    return _build_vit("dinov2_vits14_reg", pretrained=pretrained, **kwargs)


def dinov2_vitb14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-B/14 (no register tokens)."""
    return _build_vit("dinov2_vitb14", pretrained=pretrained, **kwargs)


def dinov2_vitb14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-B/14 with 4 register tokens."""
    return _build_vit("dinov2_vitb14_reg", pretrained=pretrained, **kwargs)


def dinov2_vitl14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-L/14 (no register tokens)."""
    return _build_vit("dinov2_vitl14", pretrained=pretrained, **kwargs)


def dinov2_vitl14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-L/14 with 4 register tokens."""
    return _build_vit("dinov2_vitl14_reg", pretrained=pretrained, **kwargs)


def dinov2_vitg14(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-g/14 (no register tokens)."""
    return _build_vit("dinov2_vitg14", pretrained=pretrained, **kwargs)


def dinov2_vitg14_reg(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """DINOv2 ViT-g/14 with 4 register tokens."""
    return _build_vit("dinov2_vitg14_reg", pretrained=pretrained, **kwargs)


def dinov3_vits16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-S/16 (LVD-1689M)."""
    return _build_vit(
        "dinov3_vits16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vits16plus_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-S/16+ with SwiGLU FFN (LVD-1689M)."""
    return _build_vit(
        "dinov3_vits16plus_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vitb16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-B/16 (LVD-1689M)."""
    return _build_vit(
        "dinov3_vitb16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vitl16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-L/16 (LVD-1689M)."""
    return _build_vit(
        "dinov3_vitl16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vith16plus_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-H/16+ with SwiGLU FFN (LVD-1689M)."""
    return _build_vit(
        "dinov3_vith16plus_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vit7b16_pretrain_lvd1689m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-7B/16 with SwiGLU FFN (LVD-1689M)."""
    return _build_vit(
        "dinov3_vit7b16_pretrain_lvd1689m", pretrained=pretrained, **kwargs
    )


def dinov3_vitl16_pretrain_sat493m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-L/16 (SAT-493M)."""
    return _build_vit("dinov3_vitl16_pretrain_sat493m", pretrained=pretrained, **kwargs)


def dinov3_vit7b16_pretrain_sat493m(
    pretrained: bool = False, **kwargs
) -> VisionTransformer:
    """DINOv3 ViT-7B/16 with SwiGLU FFN (SAT-493M)."""
    return _build_vit(
        "dinov3_vit7b16_pretrain_sat493m", pretrained=pretrained, **kwargs
    )


def eupe_vitt16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """EUPE ViT-T/16 (vit_tiny)."""
    return _build_vit("eupe_vitt16", pretrained=pretrained, **kwargs)


def eupe_vits16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """EUPE ViT-S/16 (vit_small)."""
    return _build_vit("eupe_vits16", pretrained=pretrained, **kwargs)


def eupe_vitb16(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """EUPE ViT-B/16 (vit_base)."""
    return _build_vit("eupe_vitb16", pretrained=pretrained, **kwargs)


def siglip2_vitb16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 224*224."""
    return _build_vit("siglip2_vitb16_224", pretrained=pretrained, **kwargs)


def siglip2_vitb16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 256*256."""
    return _build_vit("siglip2_vitb16_256", pretrained=pretrained, **kwargs)


def siglip2_vitb16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 384*384."""
    return _build_vit("siglip2_vitb16_384", pretrained=pretrained, **kwargs)


def siglip2_vitb16_512(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/16 at 512*512."""
    return _build_vit("siglip2_vitb16_512", pretrained=pretrained, **kwargs)


def siglip2_vitb32_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-B/32 at 256*256."""
    return _build_vit("siglip2_vitb32_256", pretrained=pretrained, **kwargs)


def siglip2_vitl16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-L/16 at 256*256."""
    return _build_vit("siglip2_vitl16_256", pretrained=pretrained, **kwargs)


def siglip2_vitl16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-L/16 at 384*384."""
    return _build_vit("siglip2_vitl16_384", pretrained=pretrained, **kwargs)


def siglip2_vitl16_512(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-L/16 at 512*512."""
    return _build_vit("siglip2_vitl16_512", pretrained=pretrained, **kwargs)


def siglip2_vitso400m14_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/14 at 224*224."""
    return _build_vit("siglip2_vitso400m14_224", pretrained=pretrained, **kwargs)


def siglip2_vitso400m14_378(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/14 at 378*378."""
    return _build_vit("siglip2_vitso400m14_378", pretrained=pretrained, **kwargs)


def siglip2_vitso400m16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/16 at 256*256."""
    return _build_vit("siglip2_vitso400m16_256", pretrained=pretrained, **kwargs)


def siglip2_vitso400m16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/16 at 384*384."""
    return _build_vit("siglip2_vitso400m16_384", pretrained=pretrained, **kwargs)


def siglip2_vitso400m16_512(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-SO400M/16 at 512*512."""
    return _build_vit("siglip2_vitso400m16_512", pretrained=pretrained, **kwargs)


def siglip2_vitgiantopt16_256(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-giantopt/16 at 256*256."""
    return _build_vit("siglip2_vitgiantopt16_256", pretrained=pretrained, **kwargs)


def siglip2_vitgiantopt16_384(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """SigLIP2 ViT-giantopt/16 at 384*384."""
    return _build_vit("siglip2_vitgiantopt16_384", pretrained=pretrained, **kwargs)


def tips_vits14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-S/14 high-res (448*448)."""
    return _build_vit("tips_vits14_hr", pretrained=pretrained, **kwargs)


def tips_vitb14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-B/14 high-res (448*448)."""
    return _build_vit("tips_vitb14_hr", pretrained=pretrained, **kwargs)


def tips_vitl14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-L/14 high-res (448*448)."""
    return _build_vit("tips_vitl14_hr", pretrained=pretrained, **kwargs)


def tips_vitso400m14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-SO400M/14 high-res (448*448)."""
    return _build_vit("tips_vitso400m14_hr", pretrained=pretrained, **kwargs)


def tips_vitg14_lr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-g/14 low-res (224*224)."""
    return _build_vit("tips_vitg14_lr", pretrained=pretrained, **kwargs)


def tips_vitg14_hr(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """TIPS ViT-g/14 high-res (448*448)."""
    return _build_vit("tips_vitg14_hr", pretrained=pretrained, **kwargs)
