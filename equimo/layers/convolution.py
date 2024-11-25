from typing import Callable, Optional

import equinox as eqx
import jax
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.dropout import DropPathAdd
from equimo.layers.norm import LayerScale
from equimo.utils import nearest_power_of_2_divisor


class ConvBlock(eqx.Module):
    """A residual convolutional block with normalization and regularization.

    This block implements a residual connection with two convolution layers,
    group normalization, activation, layer scaling, and drop path regularization.
    The block maintains the input dimension while allowing for an optional
    intermediate hidden dimension.

    Attributes:
        conv1: First convolution layer
        conv2: Second convolution layer
        norm1: Group normalization after first conv
        norm2: Group normalization after second conv
        drop_path1: Drop path regularization for residual connection
        act: Activation function
        ls1: Layer scaling module
    """

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    norm1: eqx.Module
    norm2: eqx.Module
    drop_path1: DropPathAdd
    act: Callable
    ls1: LayerScale

    def __init__(
        self,
        dim: int,
        *,
        key: PRNGKeyArray,
        hidden_dim: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        act_layer: Callable = jax.nn.gelu,
        norm_max_group: int = 32,
        drop_path: float = 0.0,
        init_values: float | None = None,
        **kwargs,
    ):
        """Initialize the ConvBlock.

        Args:
            dim: Input and output channel dimension
            key: PRNG key for initialization
            hidden_dim: Optional intermediate channel dimension (defaults to dim)
            kernel_size: Size of the convolutional kernel (default: 3)
            stride: Stride of the convolution (default: 1)
            padding: Padding size for convolution (default: 1)
            act_layer: Activation function (default: gelu)
            norm_max_group: Maximum number of groups for GroupNorm (default: 32)
            drop_path: Drop path rate (default: 0.0)
            init_values: Initial value for layer scaling (default: None)
            **kwargs: Additional arguments passed to Conv layers
        """

        key_conv1, key_conv2 = jr.split(key, 2)
        hidden_dim = hidden_dim or dim
        num_groups = nearest_power_of_2_divisor(dim, norm_max_group)
        self.conv1 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=True,
            key=key_conv1,
        )
        self.norm1 = eqx.nn.GroupNorm(num_groups, hidden_dim)
        self.act = act_layer
        self.conv2 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=True,
            key=key_conv2,
        )
        self.norm2 = eqx.nn.GroupNorm(num_groups, dim)

        dpr = drop_path[0] if isinstance(drop_path, list) else float(drop_path)
        self.drop_path1 = DropPathAdd(dpr)

        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else eqx.nn.Identity()
        )

    def permute(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "height width channels"]:
        return rearrange(x, "c h w -> h w c")

    def depermute(
        self,
        x: Float[Array, "height width channels"],
    ) -> Float[Array, "channels height width"]:
        return rearrange(x, "h w c -> c h w")

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "channels height width"]:
        _, h, w = x.shape
        x2 = self.act(self.norm1(self.conv1(x)))
        x2 = self.norm2(self.conv2(x2))
        x2 = self.depermute(jax.vmap(jax.vmap(self.ls1))(self.permute(x2)))

        return self.drop_path1(x, x2, inference=not enable_dropout, key=key)


class SingleConvBlock(eqx.Module):
    """A basic convolution block combining convolution, normalization and activation.

    This block provides a streamlined combination of convolution, optional group
    normalization, and optional activation in a single unit. It's designed to be
    a fundamental building block for larger architectures.

    Attributes:
        conv: Convolution layer
        norm: Normalization layer (GroupNorm or Identity)
        act: Activation layer (Lambda or Identity)
    """

    conv: eqx.nn.Conv
    norm: eqx.Module
    act: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        use_norm: bool = True,
        norm_max_group: int = 32,
        act_layer: Callable | None = None,
        **kwargs,
    ):
        """Initialize the SingleConvBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            key: PRNG key for initialization
            use_norm: Whether to use group normalization (default: True)
            norm_max_group: Maximum number of groups for GroupNorm (default: 32)
            act_layer: Optional activation function (default: None)
            **kwargs: Additional arguments passed to Conv layer
        """

        num_groups = nearest_power_of_2_divisor(out_channels, norm_max_group)
        self.conv = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            key=key,
            **kwargs,
        )
        self.norm = (
            eqx.nn.GroupNorm(num_groups, out_channels)
            if use_norm
            else eqx.nn.Identity()
        )
        self.act = eqx.nn.Lambda(act_layer) if act_layer else eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "dim height width"]:
        return self.act(self.norm(self.conv(x)))


class Stem(eqx.Module):
    """Image-to-embedding stem network for vision transformers.

    This module processes raw input images into patch embeddings through a series
    of convolutional stages. It includes three main components:
    1. Initial downsampling with conv + norm + activation
    2. Residual block with two convolutions
    3. Final downsampling and channel projection

    The output is reshaped into a sequence of patch embeddings suitable for
    transformer processing.

    Attributes:
        num_patches: Total number of patches (static)
        patches_resolution: Spatial resolution of patches (static)
        conv1: Initial convolution block
        conv2: Middle residual convolution blocks
        conv3: Final convolution blocks
    """

    num_patches: int = eqx.field(static=True)
    patches_resolution: int = eqx.field(static=True)

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    conv3: eqx.nn.Conv

    def __init__(
        self,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        img_size: int = 224,
        patch_size: int = 4,
        embed_dim=96,
        **kwargs,
    ):
        """Initialize the Stem network.

        Args:
            in_channels: Number of input image channels
            key: PRNG key for initialization
            img_size: Input image size (default: 224)
            patch_size: Size of each patch (default: 4)
            embed_dim: Final embedding dimension (default: 96)
            **kwargs: Additional arguments passed to convolution blocks
        """
        self.num_patches = (img_size // patch_size) ** 2
        self.patches_resolution = [img_size // patch_size] * 2
        (
            key_conv1,
            key_conv2,
            key_conv3,
            key_conv4,
            key_conv5,
        ) = jr.split(key, 5)

        self.conv1 = SingleConvBlock(
            in_channels=in_channels,
            out_channels=embed_dim // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=False,
            use_norm=True,
            act_layer=jax.nn.relu,
            key=key_conv1,
        )

        self.conv2 = eqx.nn.Sequential(
            [
                SingleConvBlock(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bias=False,
                    use_norm=True,
                    act_layer=jax.nn.relu,
                    key=key_conv2,
                ),
                SingleConvBlock(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_bias=False,
                    use_norm=True,
                    act_layer=None,
                    key=key_conv3,
                ),
            ]
        )

        self.conv3 = eqx.nn.Sequential(
            [
                SingleConvBlock(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    use_bias=False,
                    use_norm=True,
                    act_layer=jax.nn.relu,
                    key=key_conv4,
                ),
                SingleConvBlock(
                    in_channels=embed_dim * 4,
                    out_channels=embed_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    use_bias=False,
                    use_norm=True,
                    act_layer=None,
                    key=key_conv5,
                ),
            ]
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seqlen dim"]:
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)

        return rearrange(x, "c h w -> (h w) c")
