from typing import Callable, List, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.attention import HATBlock, WindowedAttention
from equimo.layers.convolution import ConvBlock
from equimo.layers.downsample import Downsampler
from equimo.layers.ffn import Mlp
from equimo.layers.patch import ConvPatchEmbed
from equimo.layers.sharing import LayerSharing
from equimo.utils import pool_sd, to_list


class LayerSharingWithCT(LayerSharing):
    """Layer sharing implementation that handles carrier tokens (CT).

    Extends LayerSharing to support passing and updating carrier tokens through
    the network layers while maintaining layer sharing functionality.
    """

    def __call__(
        self,
        x: Array,
        ct: Array,
        # *args,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ):
        """Apply layer sharing with carrier token support.

        Args:
            x: Input tensor
            ct: carrier token tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Tuple of (processed tensor, updated carrier token)
        """
        if self.repeat == 1:
            return self.f(
                x,
                ct,
                # *args,
                inference=inference,
                key=key,
                **kwargs,
            )

        keys = jr.split(key, self.repeat)
        reshape = len(x.shape) == 3

        for i in range(self.repeat):
            if reshape:
                _, h, w = x.shape
                lora_x = rearrange(x, "c h w -> (h w) c")
            else:
                lora_x = x
            lora_output = self.dropouts[i](
                jax.vmap(self.loras[i])(lora_x),
                inference=inference,
                key=keys[i],
            )
            if reshape:
                lora_output = rearrange(
                    lora_output,
                    "(h w) c -> c h w",
                    h=h,
                    w=w,
                )

            x, ct = self.f(
                x,
                ct,
                inference=inference,
                key=key,
                **kwargs,
            )

            x += lora_output

        return x, ct


class TokenInitializer(eqx.Module):
    """Initializes and processes carrier tokens for the network.

    Creates carrier tokens by applying positional embeddings and pooling
    to the input features. Used to generate global tokens for attention.
    """

    ct_size: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    kernel: int = eqx.field(static=True)

    pos_embed: eqx.nn.Conv
    pool: eqx.nn.AvgPool2d

    def __init__(
        self,
        dim: int,
        input_resolution: int,
        window_size: int,
        *,
        key: PRNGKeyArray,
        ct_size: int = 1,
        **kwargs,
    ):
        self.ct_size = ct_size
        output_size = int(ct_size * input_resolution / window_size)
        self.stride = int(input_resolution / output_size)
        self.kernel = int(input_resolution - (output_size - 1) * self.stride)

        self.pos_embed = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            key=key,
        )
        self.pool = eqx.nn.AvgPool2d(kernel_size=self.kernel, stride=self.stride)

    def __call__(
        self,
        x: Float[Array, "channels height weight"],
    ) -> Float[Array, "new_channels seqlen"]:
        x = self.pos_embed(x)
        x = self.pool(x)

        ct = rearrange(
            x, "c (h h1) (w w1) -> (h w h1 w1) c", h1=self.ct_size, w1=self.ct_size
        )

        return ct


class BlockChunk(eqx.Module):
    """A chunk of processing blocks with optional downsampling and global tokenization.

    Handles a sequence of processing blocks (either ConvBlock or HATBlock),
    with support for window partitioning, global tokenization, and downsampling.
    """

    reshape: bool = eqx.field(static=True)
    downsampler_contains_dropout: bool = eqx.field(static=True)
    is_hat: bool = eqx.field(static=True)
    ct_size: int = eqx.field(static=True)
    window_size: bool = eqx.field(static=True)
    do_gt: bool = eqx.field(static=True)

    blocks: List[eqx.Module]
    downsample: eqx.Module
    global_tokenizer: Optional[TokenInitializer]

    def __init__(
        self,
        input_resolution: int,
        window_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
        block: eqx.Module = HATBlock,
        repeat: int = 1,
        downsampler: eqx.Module = Downsampler,
        downsampler_contains_dropout: bool = False,
        only_local: bool = False,
        hierarchy: bool = True,
        ct_size: int = 1,
        **kwargs,
    ):
        key_ds, key_gt, *block_subkeys = jr.split(key, depth + 2)
        if not isinstance(downsampler, eqx.nn.Identity):
            if kwargs.get("dim") is None:
                raise ValueError(
                    "Using a downsampler requires passing a `dim` argument."
                )

        # self.reshape = block is not ConvBlock
        self.reshape = True  # TODO
        self.downsampler_contains_dropout = downsampler_contains_dropout
        self.is_hat = block is HATBlock
        self.ct_size = ct_size
        self.window_size = window_size

        keys_to_spread = [
            k for k, v in kwargs.items() if isinstance(v, list) and len(v) == depth
        ]

        self.blocks = []
        for i in range(depth):
            config = kwargs | {k: kwargs[k][i] for k in keys_to_spread}

            if self.is_hat:
                config = config | {
                    "sr_ratio": (
                        input_resolution // window_size if not only_local else 1
                    ),
                    "window_size": window_size,
                    "last": i == depth - 1,
                    "ct_size": self.ct_size,
                }

            wrapper = LayerSharingWithCT if self.is_hat else LayerSharing
            self.blocks.append(
                wrapper(
                    dim=kwargs.get("dim"),
                    f=block(**config, key=block_subkeys[i]),
                    repeat=repeat,
                    key=block_subkeys[i],
                ),
            )

        self.downsample = downsampler(dim=kwargs.get("dim"), key=key_ds)

        if (
            len(self.blocks)
            and not only_local
            and input_resolution // window_size > 1
            and hierarchy
            and self.is_hat
        ):
            self.do_gt = True
            self.global_tokenizer = TokenInitializer(
                kwargs.get("dim"),
                input_resolution,
                window_size,
                ct_size=self.ct_size,
                key=key_gt,
            )
        else:
            self.do_gt = False
            self.global_tokenizer = eqx.nn.Identity()

    def window_partition(
        self,
        x: Float[Array, "channels height width"],
        window_size: int,
    ) -> Float[Array, "patches window_size channels"]:
        """Partition input tensor into windows.

        Args:
            x: Input tensor of shape (channels, height, width)
            window_size: Size of each window

        Returns:
            Tensor partitioned into windows of specified size
        """
        return rearrange(
            x,
            "c (h h1) (w w1) -> (h w) (h1 w1) c",
            h1=window_size,
            w1=window_size,
        )

    def window_reverse(
        self,
        x: Float[Array, "patches window_size channels"],
        window_size: int,
        h: int,
        w: int,
    ) -> Float[Array, "channels height width"]:
        """Reverse window partitioning to original tensor shape.

        Args:
            x: Window partitioned tensor
            window_size: Size of each window
            h: Original height
            w: Original width

        Returns:
            Tensor restored to original spatial dimensions
        """
        return rearrange(
            x,
            "(h w) (h1 w1) c -> c (h h1) (w w1)",
            h=h // window_size,
            w=w // window_size,
            h1=window_size,
            w1=window_size,
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        keys = jr.split(key, len(self.blocks))
        ct = self.global_tokenizer(x) if self.do_gt else None
        c, h, w = x.shape

        if self.is_hat:
            x = self.window_partition(x, self.window_size)
            for blk, key_block in zip(self.blocks, keys):
                x, ct = jax.vmap(
                    blk,
                    in_axes=(0, None, None, None),
                    out_axes=(0, None),
                )(x, ct, inference=inference, key=key_block)
            x = self.window_reverse(x, self.window_size, h, w)
        else:
            for blk, key_block in zip(self.blocks, keys):
                x = blk(x, inference=inference, key=key_block)

        if self.downsampler_contains_dropout:
            x = self.downsample(x, inference=inference, key=key)
        else:
            x = self.downsample(x)

        return x


class FasterViT(eqx.Module):
    """FasterViT: Fast Vision Transformers with Hierarchical Attention[1]

    Implements a vision transformer architecture that combines convolution and
    hierarchical attention mechanisms for efficient image processing. Features
    carrier tokens, window attention, and progressive feature resolution reduction.

    References:
        [1]: Hatamizadeh, et al., 2023. https://arxiv.org/abs/2306.06189
    """

    patch_embed: ConvPatchEmbed
    blocks: List[eqx.Module]
    norm: eqx.Module
    head: eqx.Module

    dim: int = eqx.field(static=True)
    global_pool: str = eqx.field(static=True)

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        dim: int,
        in_dim: int,
        num_heads: int | List[int],
        hat: bool | List[bool],
        depths: List[int],
        window_size: int | List[int],
        ct_size: int,
        *,
        key: PRNGKeyArray,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        block: eqx.Module = HATBlock,
        repeat: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = jax.nn.gelu,
        attn_layer: eqx.Module = WindowedAttention,
        ffn_layer: eqx.Module = Mlp,
        ffn_bias: bool = True,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        init_values: float | None = 1e-5,
        ls_convblock: bool = False,
        do_propagation: bool = False,
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
        self.global_pool = global_pool

        self.patch_embed = ConvPatchEmbed(
            in_channels,
            in_dim,
            dim,
            key=key_patchemb,
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = list(jnp.linspace(0.0, drop_path_rate, depth))

        n_chunks = len(depths)
        num_heads = to_list(num_heads, n_chunks)
        hat = to_list(hat, n_chunks)
        attn_layer = to_list(attn_layer, n_chunks)
        window_size = to_list(window_size, n_chunks)
        self.blocks = [
            BlockChunk(
                block=ConvBlock if i < 2 else HATBlock,
                repeat=repeat,
                dim=int(dim * 2**i),
                depth=depths[i],
                input_resolution=int(2 ** (-2 - i) * img_size),
                window_size=window_size[i],
                ct_size=ct_size,
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
                init_values=None if i < 2 and not ls_convblock else init_values,
                downsampler=Downsampler if i < len(depths) - 1 else eqx.nn.Identity,
                only_local=not hat[i],
                do_propagation=do_propagation,
                key=block_subkeys[i],
            )
            for i, depth in enumerate(depths)
        ]

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.norm = norm_layer(num_features)
        self.head = (
            eqx.nn.Linear(num_features, num_classes, key=key_head)
            if num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        """Extract features from input image.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Processed feature tensor
        """
        key_posdrop, *block_subkeys = jr.split(key, len(self.blocks) + 1)
        x = self.patch_embed(x)

        for blk, key_block in zip(self.blocks, block_subkeys):
            x = blk(x, inference=inference, key=key_block)

        x = rearrange(x, "c h w -> (h w) c")

        return x

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "num_classes"]:
        """Process input image through the full network.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Classification logits for each class
        """
        x = self.features(x, inference=inference, key=key)
        x = jax.vmap(self.norm)(x)
        x = pool_sd(
            x,
            num_prefix_tokens=0,
            pool_type=self.global_pool,
            reduce_include_prefix=False,
        )

        x = self.head(x)

        return x
