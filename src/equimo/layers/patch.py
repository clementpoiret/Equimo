import math
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.convolution import SingleConvBlock
from equimo.layers.squeeze_excite import SEModule
from equimo.utils import make_2tuple, nearest_power_of_2_divisor


class PatchEmbedding(eqx.Module):
    """Image to patch embedding module for vision transformers.

    This module converts an image into a sequence of patch embeddings by:
    1. Splitting the image into fixed-size patches
    2. Projecting each patch to an embedding dimension
    3. Optionally flattening the spatial dimensions

    Supports dynamic image sizes and padding when needed.

    Attributes:
        patch_size: Size of each patch (static)
        img_size: Input image size (static)
        grid_size: Grid dimensions after patching (static)
        num_patches: Total number of patches (static)
        flatten: Whether to flatten spatial dimensions (static)
        dynamic_img_size: Allow variable image sizes (static)
        dynamic_img_pad: Allow padding for non-divisible sizes (static)
        proj: Patch projection layer
        norm: Normalization layer
    """

    patch_size: int | Tuple[int, int] = eqx.field(static=True)

    img_size: Optional[Tuple[int, int]] = eqx.field(static=True)
    grid_size: Optional[Tuple[int, int]] = eqx.field(static=True)
    num_patches: Optional[int] = eqx.field(static=True)

    flatten: bool = eqx.field(static=True)
    dynamic_img_size: bool = eqx.field(static=True)
    dynamic_img_pad: bool = eqx.field(static=True)

    proj: eqx.nn.Embedding
    norm: eqx.Module

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int | Tuple[int, int],
        *,
        key: PRNGKeyArray,
        img_size: Optional[int | Tuple[int, int]] = None,
        flatten: bool = True,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        norm_layer: eqx.Module = eqx.nn.Identity,
        eps: float = 1e-5,
        **kwargs,
    ):
        patch_size = make_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        if img_size is None:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None
        else:
            self.img_size = make_2tuple(img_size)
            self.grid_size = (
                self.img_size[0] // patch_size[0],
                self.img_size[1] // patch_size[1],
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,  # tuple of 2 ints
            stride=patch_size,
            key=key,
        )
        self.norm = norm_layer(dim, eps=eps)

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1]
            )
        return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def __call__(self, x: Float[Array, "channels height width"]) -> Float[Array, "..."]:
        C, H, W = x.shape

        if self.img_size is not None:
            if not self.dynamic_img_size:
                if H != self.img_size[0]:
                    raise AssertionError(
                        f"Input height ({H}) doesn't match model ({self.img_size[0]})"
                    )
                if W != self.img_size[1]:
                    raise AssertionError(
                        f"Input width ({W}) doesn't match model ({self.img_size[1]})"
                    )
            elif not self.dynamic_img_pad:
                if H % self.patch_size[0] != 0:
                    raise AssertionError(
                        f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})"
                    )
                if W % self.patch_size[1] != 0:
                    raise AssertionError(
                        f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})"
                    )

        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = jnp.pad(x, pad_width=((0, 0), (0, pad_h), (0, pad_w)))

        x = self.proj(x)
        C, H, W = x.shape

        x = rearrange(x, "c h w -> (h w) c")
        x = jax.vmap(self.norm)(x)

        if not self.flatten:
            return rearrange(x, "(h w) c -> c h w", h=H, w=W)

        return x


class ConvPatchEmbed(eqx.Module):
    """
    Convolutional Patch Embedding, used in MambaVision.

    This module applies a series of convolutional layers to embed patches.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        act_layer (Callable, optional): Activation function to use. Defaults to jax.nn.relu.
        norm_layer (Callable, optional): Normalization layer to use. Defaults to eqx.nn.BatchNorm.
        key (PRNGKeyArray): Random number for PRNG.
    """

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    norm1: eqx.Module
    norm2: eqx.Module
    act: Callable

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        act_layer: Callable = jax.nn.relu,
        norm_layer: Callable = eqx.nn.LayerNorm,
        eps: float = 1e-5,
    ):
        key_conv1, key_conv2 = jr.split(key, 2)

        self.conv1 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            use_bias=False,
            key=key_conv1,
        )
        self.norm1 = norm_layer(hidden_channels, eps=eps)
        self.act = act_layer
        self.conv2 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=2,
            padding=1,
            use_bias=False,
            key=key_conv2,
        )
        self.norm2 = norm_layer(out_channels, eps=eps)

    def flatten(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "space channels"]:
        return rearrange(x, "c h w -> (h w) c")

    def deflatten(
        self,
        x: Float[Array, "space channels"],
        h: int,
        w: int,
    ) -> Float[Array, "channels height width"]:
        return rearrange(x, "(h w) c -> c h w", h=h, w=w)

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "new_channels new_height new_width"]:
        """
        Forward pass of the ConvPatchEmbed module.

        Args:
            x (jnp.ndarray): Input tensor.

        Returns:
            jnp.ndarray: Output tensor after applying convolutional patch embedding.
        """
        c, h, w = x.shape
        x = self.deflatten(
            self.act(jax.vmap(self.norm1)(self.flatten(self.conv1(x)))),
            h=h // 2,
            w=w // 2,
        )
        x = self.deflatten(
            self.act(jax.vmap(self.norm2)(self.flatten(self.conv2(x)))),
            h=h // 4,
            w=w // 4,
        )
        return x


class PatchMerging(eqx.Module):
    """Patch merging module that reduces spatial resolution while increasing channels.

    This module implements a hierarchical feature aggregation using three convolution stages:
    1. 1x1 conv to expand channels
    2. 3x3 depthwise conv with stride 2 for spatial reduction
    3. 1x1 conv to adjust final channel dimension

    The module follows an inverted bottleneck architecture with expansion ratio.

    Attributes:
        conv1: First 1x1 convolution for channel expansion
        conv2: 3x3 depthwise convolution for spatial reduction
        conv3: Final 1x1 convolution for channel adjustment
    """

    conv1: eqx.Module
    conv2: eqx.Module
    conv3: eqx.Module

    def __init__(
        self,
        dim: int,
        *,
        key: PRNGKeyArray,
        ratio: float = 4.0,
        **kwargs,
    ):
        (
            key_conv1,
            key_conv2,
            key_conv3,
        ) = jr.split(key, 3)
        out_dim = int(dim * 2)
        hidden_dim = int(out_dim * ratio)

        self.conv1 = SingleConvBlock(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=None,
            act_layer=jax.nn.relu,
            key=key_conv1,
        )
        self.conv2 = SingleConvBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=hidden_dim,
            norm_layer=None,
            act_layer=jax.nn.relu,
            key=key_conv2,
        )
        self.conv3 = SingleConvBlock(
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_layer=None,
            act_layer=jax.nn.relu,
            key=key_conv3,
        )

    def __call__(self, x: Float[Array, "seqlen dim"]) -> Float[Array, "new_seqlen dim"]:
        l, _ = x.shape
        h = w = int(l**0.5)

        x = rearrange(x, "(h w) c -> c h w", h=h, w=w)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = rearrange(x, "c h w -> (h w) c")

        return x


class SEPatchMerging(eqx.Module):
    """Squeeze-and-Excite Patch Merging module.

    This module combines patch merging with squeeze-and-excitation attention.
    The architecture consists of:
    1. Channel expansion through 1x1 conv
    2. Spatial reduction with 3x3 conv
    3. SE attention module for channel recalibration
    4. Channel projection with 1x1 conv
    Each conv layer is followed by group normalization and ReLU activation.

    Attributes:
        conv1: First 1x1 convolution layer
        conv2: 3x3 convolution layer for spatial reduction
        conv3: Final 1x1 convolution layer
        norm1: Group normalization after first conv
        norm2: Group normalization after second conv
        norm3: Group normalization after third conv
        se: Squeeze-and-Excitation module
    """

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    conv3: eqx.nn.Conv
    norm1: eqx.Module
    norm2: eqx.Module
    norm3: eqx.Module
    se: SEModule

    def __init__(
        self,
        dim: int,
        out_dim: int,
        *,
        key: PRNGKeyArray,
        norm_max_group: int = 32,
        **kwargs,
    ):
        key_conv1, key_conv2, key_conv3, key_se = jr.split(key, 4)
        hidden_dim = int(dim * 4)
        num_groups = nearest_power_of_2_divisor(hidden_dim, norm_max_group)

        self.conv1 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv1,
        )
        self.norm1 = eqx.nn.GroupNorm(num_groups, hidden_dim)
        self.conv2 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            key=key_conv2,
        )
        self.norm2 = eqx.nn.GroupNorm(num_groups, hidden_dim)
        self.se = SEModule(hidden_dim, rd_ratio=1.0 / 4, key=key_se)
        self.conv3 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv3,
        )
        self.norm3 = eqx.nn.GroupNorm(
            nearest_power_of_2_divisor(out_dim, norm_max_group),
            out_dim,
        )

    def __call__(self, x: Float[Array, "channels height width"]) -> Float[Array, "..."]:
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        x = jax.nn.relu(self.norm2(self.conv2(x)))
        x = self.se(x)
        x = self.norm3(self.conv3(x))

        return x
