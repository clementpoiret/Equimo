import math
import operator
from typing import Callable, Literal, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange, reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.dropout import DropPathAdd
from equimo.layers.norm import LayerScale
from equimo.utils import make_divisible, nearest_power_of_2_divisor


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
    ls1: LayerScale | None

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
        num_groups1 = nearest_power_of_2_divisor(hidden_dim, norm_max_group)
        num_groups2 = nearest_power_of_2_divisor(dim, norm_max_group)
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
        self.norm1 = eqx.nn.GroupNorm(num_groups1, hidden_dim)
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
        self.norm2 = eqx.nn.GroupNorm(num_groups2, dim)

        dpr = drop_path[0] if isinstance(drop_path, list) else float(drop_path)
        self.drop_path1 = DropPathAdd(dpr)

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else None

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
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "channels height width"]:
        x2 = self.act(self.norm1(self.conv1(x)))
        x2 = self.norm2(self.conv2(x2))
        if self.ls1 is not None:
            x2 = self.depermute(jax.vmap(jax.vmap(self.ls1))(self.permute(x2)))

        return self.drop_path1(x, x2, inference=inference, key=key)


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

    conv: eqx.nn.Conv2d | eqx.nn.ConvTranspose2d
    norm: eqx.Module
    act: eqx.Module
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = "SAME",
        norm_layer: eqx.Module | None = eqx.nn.GroupNorm,
        norm_max_group: int = 32,
        act_layer: Callable | None = None,
        dropout: float = 0.0,
        transposed: bool = False,
        norm_kwargs: dict = {},
        **kwargs,
    ):
        """Initialize the SingleConvBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            key: PRNG key for initialization
            norm_max_group: Maximum number of groups for GroupNorm (default: 32)
            act_layer: Optional activation function (default: None)
            norm_kwargs: Args passed to the norm layer. This allows disabling
                weights of LayerNorm, which do not work well with conv layers
            **kwargs: Additional arguments passed to Conv layer
        """

        conv = eqx.nn.ConvTranspose2d if transposed else eqx.nn.Conv2d
        self.conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            key=key,
            **kwargs,
        )

        # TODO: test
        if norm_layer is not None:
            if norm_layer == eqx.nn.GroupNorm:
                num_groups = nearest_power_of_2_divisor(out_channels, norm_max_group)
                self.norm = eqx.nn.GroupNorm(num_groups, out_channels, **norm_kwargs)
            else:
                self.norm = norm_layer(out_channels, **norm_kwargs)
        else:
            self.norm = eqx.nn.Identity()

        self.dropout = eqx.nn.Dropout(dropout)
        self.act = eqx.nn.Lambda(act_layer) if act_layer else eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "dim height width"]:
        return self.dropout(
            self.act(self.norm(self.conv(x))), inference=inference, key=key
        )


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


class ConvBottleneck(eqx.Module):
    """YOLO's Bottleneck to be used into a C2F or C3k2 block."""

    add: bool = eqx.field(static=True)

    conv1: SingleConvBlock
    conv2: SingleConvBlock

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        shortcut: bool = True,
        groups: int = 1,
        kernel_sizes: Sequence[int] = [3, 3],
        expansion_ratio: float = 0.5,
    ):
        key_conv1, key_conv2 = jr.split(key, 2)

        hidden_channels = int(out_channels * expansion_ratio)

        self.conv1 = SingleConvBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            act_layer=jax.nn.silu,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding="SAME",
            key=key_conv1,
        )
        self.conv2 = SingleConvBlock(
            in_channels=hidden_channels,
            out_channels=out_channels,
            act_layer=jax.nn.silu,
            kernel_size=kernel_sizes[1],
            stride=1,
            padding="SAME",
            groups=groups,
            key=key_conv2,
        )

        self.add = shortcut and in_channels == out_channels

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        x1 = self.conv2(self.conv1(x))

        if self.add:
            return x + x1
        return x1


class C2f(eqx.Module):
    """YOLO's Fast CSP Bottleneck"""

    hidden_channels: int = eqx.field(static=True)

    conv1: SingleConvBlock
    conv2: SingleConvBlock
    blocks: list[ConvBottleneck]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        n: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion_ratio: float = 0.5,
    ):
        key_conv1, key_conv2, *key_blocks = jr.split(key, 2 + n)

        self.hidden_channels = int(out_channels * expansion_ratio)

        self.conv1 = SingleConvBlock(
            in_channels=in_channels,
            out_channels=self.hidden_channels * 2,
            act_layer=jax.nn.silu,
            kernel_size=1,
            stride=1,
            padding="SAME",
            key=key_conv1,
        )
        self.conv2 = SingleConvBlock(
            in_channels=(2 + n) * self.hidden_channels,
            out_channels=out_channels,
            act_layer=jax.nn.silu,
            kernel_size=1,
            stride=1,
            padding="SAME",
            key=key_conv2,
        )

        self.blocks = [
            ConvBottleneck(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                shortcut=shortcut,
                groups=groups,
                kernel_sizes=[3, 3],
                expansion_ratio=1.0,
                key=key_blocks[i],
            )
            for i in range(n)
        ]

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        y = jnp.split(self.conv1(x), [self.hidden_channels])
        y.extend(blk(y[-1]) for blk in self.blocks)
        return self.conv2(jnp.concatenate(y, axis=0))


class C3k(eqx.Module):
    """YOLO's Fast CSP Bottleneck with 3 convolutions with customizable kernel"""

    conv1: SingleConvBlock
    conv2: SingleConvBlock
    conv3: SingleConvBlock
    blocks: eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        n: int = 1,
        kernel_sizes: Sequence[int] = [3, 3],
        shortcut: bool = True,
        groups: int = 1,
        expansion_ratio: float = 0.5,
    ):
        key_conv1, key_conv2, key_conv3, *key_blocks = jr.split(key, 3 + n)

        hidden_channels = int(out_channels * expansion_ratio)

        self.conv1 = SingleConvBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            act_layer=jax.nn.silu,
            kernel_size=1,
            stride=1,
            padding="SAME",
            key=key_conv1,
        )
        self.conv2 = SingleConvBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            act_layer=jax.nn.silu,
            kernel_size=1,
            stride=1,
            padding="SAME",
            key=key_conv2,
        )
        self.conv3 = SingleConvBlock(
            in_channels=2 * hidden_channels,
            out_channels=out_channels,
            act_layer=jax.nn.silu,
            kernel_size=1,
            stride=1,
            padding="SAME",
            key=key_conv3,
        )

        self.blocks = eqx.nn.Sequential(
            [
                ConvBottleneck(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    shortcut=shortcut,
                    groups=groups,
                    kernel_sizes=kernel_sizes,
                    expansion_ratio=1.0,
                    key=key_blocks[i],
                )
                for i in range(n)
            ]
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        return self.conv3(
            jnp.concatenate([self.blocks(self.conv1(x)), self.conv2(x)], axis=0)
        )


class C3(eqx.Module):
    """YOLO's Fast CSP Bottleneck with 3 convolutions"""

    c3k: C3k

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        n: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion_ratio: float = 0.5,
    ):
        self.c3k = C3k(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=[1, 3],
            shortcut=shortcut,
            groups=groups,
            expansion_ratio=expansion_ratio,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        return self.c3k(x)


class C3k2(eqx.Module):
    """YOLO's Fast CSP Bottleneck"""

    hidden_channels: int = eqx.field(static=True)

    conv1: SingleConvBlock
    conv2: SingleConvBlock
    blocks: list[ConvBottleneck] | list[C3k]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        n: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion_ratio: float = 0.5,
        c3k: bool = True,
    ):
        key_conv1, key_conv2, *key_blocks = jr.split(key, 2 + n)

        self.hidden_channels = int(out_channels * expansion_ratio)

        self.conv1 = SingleConvBlock(
            in_channels=in_channels,
            out_channels=self.hidden_channels * 2,
            act_layer=jax.nn.silu,
            kernel_size=1,
            stride=1,
            padding="SAME",
            key=key_conv1,
        )
        self.conv2 = SingleConvBlock(
            in_channels=(2 + n) * self.hidden_channels,
            out_channels=out_channels,
            act_layer=jax.nn.silu,
            kernel_size=1,
            stride=1,
            padding="SAME",
            key=key_conv2,
        )

        if c3k:
            self.blocks = [
                C3k(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    n=2,
                    shortcut=shortcut,
                    groups=groups,
                    key=key_blocks[i],
                )
                for i in range(n)
            ]
        else:
            self.blocks = [
                ConvBottleneck(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    shortcut=shortcut,
                    groups=groups,
                    key=key_blocks[i],
                )
                for i in range(n)
            ]

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        y = jnp.split(self.conv1(x), [self.hidden_channels])
        y.extend(blk(y[-1]) for blk in self.blocks)
        return self.conv2(jnp.concatenate(y, axis=0))


class MBConv(eqx.Module):
    """MobileNet Conv Block with optional fusing from [1].

    References:
        [1]: Nottebaum, M., Dunnhofer, M., & Micheloni, C. (2024). LowFormer:
        Hardware Efficient Design for Convolutional Transformer Backbones (No.
        arXiv:2409.03460). arXiv. https://doi.org/10.48550/arXiv.2409.03460
    """

    fused: bool = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    inverted_conv: SingleConvBlock | None
    depth_conv: SingleConvBlock | None
    spatial_conv: SingleConvBlock | None
    point_conv: SingleConvBlock
    drop_path: DropPathAdd

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        mid_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: Tuple[bool, ...] | bool = False,
        expand_ratio: float = 6.0,
        norm_layers: Tuple[eqx.Module | None, ...]
        | eqx.Module
        | None = eqx.nn.GroupNorm,
        act_layers: Tuple[Callable | None, ...] | Callable | None = jax.nn.relu6,
        fuse: bool = False,
        fuse_threshold: int = 256,
        fuse_group: bool = False,
        fused_conv_groups: int = 1,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        residual: bool = False,
        **kwargs,
    ):
        key_inverted, key_depth, key_point = jr.split(key, 3)

        if not isinstance(norm_layers, Tuple):
            norm_layers = (norm_layers,) * 3
        if not isinstance(act_layers, Tuple):
            act_layers = (act_layers,) * 3
        if isinstance(use_bias, bool):
            use_bias: Tuple = (use_bias,) * 3
        if len(use_bias) != 3:
            raise ValueError(
                f"`use_bias` should be a Tuple of length 3, got: {len(use_bias)}"
            )
        if len(norm_layers) != 3:
            raise ValueError(
                f"`norm_layers` should be a Tuple of length 3, got: {len(norm_layers)}"
            )
        if len(act_layers) != 3:
            raise ValueError(
                f"`act_layers` should be a Tuple of length 3, got: {len(act_layers)}"
            )

        # Ensure shapes are the same between input and output
        self.residual = residual and (stride == 1) and (in_channels == out_channels)

        mid_channels = (
            mid_channels
            if mid_channels is not None
            else round(in_channels * expand_ratio)
        )
        self.fused = fuse and in_channels <= fuse_threshold

        self.inverted_conv = (
            SingleConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layers[0],
                act_layer=act_layers[0],
                use_bias=use_bias[0],
                padding="SAME",
                key=key_inverted,
            )
            if not self.fused
            else None
        )
        self.depth_conv = (
            SingleConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=mid_channels,
                norm_layer=norm_layers[1],
                act_layer=act_layers[1],
                use_bias=use_bias[1],
                padding="SAME",
                key=key_depth,
            )
            if not self.fused
            else None
        )
        self.spatial_conv = (
            SingleConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=2
                if fuse_group and fused_conv_groups == 1
                else fused_conv_groups,
                norm_layer=norm_layers[0],
                act_layer=act_layers[0],
                use_bias=use_bias[0],
                padding="SAME",
                key=key_depth,
            )
            if self.fused
            else None
        )
        self.point_conv = SingleConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layers[2],
            act_layer=act_layers[2],
            use_bias=use_bias[2],
            padding="SAME",
            dropout=dropout,
            key=key_point,
        )

        self.drop_path = DropPathAdd(drop_path)

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ):
        key_spatial, key_inverted, key_depth, key_point, key_droppath = jr.split(key, 5)
        if self.fused:
            out = self.spatial_conv(x, inference=inference, key=key_spatial)
        else:
            out = self.inverted_conv(x, inference=inference, key=key_inverted)
            out = self.depth_conv(out, inference=inference, key=key_depth)
        out = self.point_conv(out, inference=inference, key=key_point)

        if self.residual:
            out = self.drop_path(x, out, inference=inference, key=key_droppath)

        return out


class DSConv(eqx.Module):
    residual: bool = eqx.field(static=True)

    depth_conv: SingleConvBlock
    point_conv: SingleConvBlock
    dropout: eqx.nn.Dropout
    drop_path: DropPathAdd

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: Tuple[bool, ...] | bool = False,
        norm_layers: Tuple[eqx.Module | None, ...]
        | eqx.Module
        | None = eqx.nn.GroupNorm,
        act_layers: Tuple[Callable | None, ...] | Callable | None = jax.nn.relu6,
        residual: bool = False,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        **kwargs,
    ):
        key_depth, key_point = jr.split(key, 2)

        if not isinstance(norm_layers, Tuple):
            norm_layers = (norm_layers,) * 2
        if not isinstance(act_layers, Tuple):
            act_layers = (act_layers,) * 2
        if isinstance(use_bias, bool):
            use_bias: Tuple = (use_bias,) * 2
        if len(use_bias) != 2:
            raise ValueError(
                f"`use_bias` should be a Tuple of length 2, got: {len(use_bias)}"
            )
        if len(norm_layers) != 2:
            raise ValueError(
                f"`norm_layers` should be a Tuple of length 2, got: {len(norm_layers)}"
            )
        if len(act_layers) != 2:
            raise ValueError(
                f"`act_layers` should be a Tuple of length 2, got: {len(act_layers)}"
            )

        # Ensure shapes are the same between input and output
        self.residual = residual and (stride == 1) and (in_channels == out_channels)

        self.depth_conv = SingleConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            norm_layer=norm_layers[0],
            act_layer=act_layers[0],
            use_bias=use_bias[0],
            padding="SAME",
            key=key_depth,
        )
        self.point_conv = SingleConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layers[1],
            act_layer=act_layers[1],
            use_bias=use_bias[1],
            padding="SAME",
            key=key_point,
        )

        self.dropout = eqx.nn.Dropout(dropout)
        self.drop_path = DropPathAdd(drop_path)

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ):
        key_depth, key_point, key_dropout, key_droppath = jr.split(key, 4)

        out = self.depth_conv(x, inference=inference, key=key_depth)
        out = self.point_conv(out, inference=inference, key=key_point)

        out = self.dropout(out, inference=inference, key=key_dropout)

        if self.residual:
            out = self.drop_path(x, out, inference=inference, key=key_droppath)

        return out


class UIB(eqx.Module):
    """MobileNet v4's Universal Inverted Bottleneck with optional fusing from [1].

    References:
        [1]: Qin, Danfeng, Chas Leichner, Manolis Delakis, Marco Fornoni,
        Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin
        Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, and Andrew Howard.
        2024. “MobileNetV4 -- Universal Models for the Mobile Ecosystem.”
    """

    residual: bool = eqx.field(static=True)

    start_dw_conv: SingleConvBlock | None
    expand_conv: SingleConvBlock
    middle_dw_conv: SingleConvBlock | None
    proj_conv: SingleConvBlock

    dropout: eqx.nn.Dropout
    drop_path: DropPathAdd

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        start_dw_kernel_size: int | None,
        middle_dw_kernel_size: int | None,
        middle_dw_downsample: bool = True,
        stride: int = 1,
        expand_ratio: float = 6.0,
        norm_layer: eqx.Module = eqx.nn.GroupNorm,
        act_layer: Callable | None = jax.nn.relu,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        residual: bool = False,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_sdwc, key_ec, key_mdwc, key_proj = jr.split(key, 4)

        self.start_dw_conv = (
            SingleConvBlock(
                in_channels,
                in_channels,
                kernel_size=start_dw_kernel_size,
                stride=stride if not middle_dw_downsample else 1,
                padding=(start_dw_kernel_size - 1) // 2,
                groups=in_channels,
                use_bias=False,
                norm_layer=norm_layer,
                key=key_sdwc,
            )
            if start_dw_kernel_size
            else None
        )

        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = SingleConvBlock(
            in_channels,
            expand_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=False,
            norm_layer=norm_layer,
            act_layer=act_layer,
            key=key_ec,
        )

        self.middle_dw_conv = (
            SingleConvBlock(
                expand_channels,
                expand_channels,
                kernel_size=middle_dw_kernel_size,
                stride=stride if middle_dw_downsample else 1,
                padding=(middle_dw_kernel_size - 1) // 2,
                groups=expand_channels,
                use_bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
                key=key_mdwc,
            )
            if middle_dw_kernel_size
            else None
        )

        self.proj_conv = SingleConvBlock(
            expand_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            use_bias=False,
            norm_layer=norm_layer,
            key=key_proj,
        )

        # Ensure shapes are the same between input and output
        self.residual = residual and (stride == 1) and (in_channels == out_channels)
        self.dropout = eqx.nn.Dropout(dropout)
        self.drop_path = DropPathAdd(drop_path)

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ):
        key_sdwc, key_ec, key_mdwc, key_proj, key_dropout, key_droppath = jr.split(
            key, 6
        )

        out = x

        if self.start_dw_conv is not None:
            out = self.start_dw_conv(out, inference=inference, key=key_sdwc)

        out = self.expand_conv(out, inference=inference, key=key_ec)

        if self.middle_dw_conv is not None:
            out = self.middle_dw_conv(out, inference=inference, key=key_mdwc)

        out = self.proj_conv(out, inference=inference, key=key_proj)

        out = self.dropout(out, inference=inference, key=key_dropout)

        if self.residual:
            out = self.drop_path(x, out, inference=inference, key=key_droppath)

        return out


class GenericGhostModule(eqx.Module):
    """Modded module for the GhostNet v3 model.


    It differs from the original implementation because it shares the same norm
    ayers accross multiple parallel branches to allow using other norm layers
    while still fusing convolutions at test-time.
    """

    mode: Literal["original", "shortcut"] = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    num_conv_branches: int = eqx.field(static=True)
    inference: bool

    primary_conv: eqx.nn.Conv2d
    cheap_operation: eqx.nn.Conv2d

    primary_rpr_conv: list[eqx.nn.Conv2d]
    primary_shared_norm: eqx.nn.GroupNorm
    primary_activation: Callable
    cheap_rpr_scale: eqx.nn.Conv2d | eqx.nn.Identity
    cheap_rpr_skip: eqx.nn.Conv2d | eqx.nn.Identity
    cheap_rpr_conv: list[eqx.nn.Conv2d]
    cheap_shared_norm: eqx.nn.GroupNorm
    cheap_activation: Callable
    short_conv: eqx.nn.Identity | eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 1,
        ratio: int = 2,
        dw_size: int = 3,
        stride: int = 1,
        act_layer: Callable = jax.nn.relu,
        num_conv_branches: int = 3,
        mode: Literal["original", "shortcut"] = "original",
        key: PRNGKeyArray,
    ):
        key_primary, key_cheap, key_crs, key_skip, key_s1, key_s2, key_s3 = jr.split(
            key, 7
        )
        key_ps = jr.split(key_primary, num_conv_branches)
        key_cs = jr.split(key_cheap, num_conv_branches)

        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.inference = False
        self.mode = mode
        self.num_conv_branches = num_conv_branches
        self.out_channels = out_channels

        # Those are actually placeholders, updated at each epoch, only used at inference time
        self.primary_conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            key=key_primary,
        )
        self.cheap_operation = eqx.nn.Conv2d(
            in_channels=init_channels,
            out_channels=new_channels,
            kernel_size=dw_size,
            stride=stride,
            padding=dw_size // 2,
            groups=1,
            key=key_cheap,
        )

        init_num_groups = nearest_power_of_2_divisor(init_channels, 32)
        # TODO: test with some dropout?
        self.primary_rpr_conv = [
            eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=init_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                key=key_ps[i],
            )
            for i in range(num_conv_branches)
        ]
        self.primary_shared_norm = eqx.nn.GroupNorm(init_num_groups, init_channels)
        self.primary_activation = act_layer

        self.cheap_rpr_scale = eqx.nn.Conv2d(
            in_channels=init_channels,
            out_channels=new_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            key=key_crs,
        )
        self.cheap_rpr_skip = eqx.nn.Conv2d(
            init_channels,
            new_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=1,
            key=key_skip,
        )
        self.cheap_rpr_conv = [
            eqx.nn.Conv2d(
                in_channels=init_channels,
                out_channels=new_channels,
                kernel_size=dw_size,
                stride=stride,
                padding=dw_size // 2,
                groups=1,
                key=key_cs[i],
            )
            for i in range(self.num_conv_branches)
        ]
        newchannels_num_groups = nearest_power_of_2_divisor(new_channels, 32)
        self.cheap_shared_norm = eqx.nn.GroupNorm(newchannels_num_groups, new_channels)
        self.cheap_activation = act_layer

        out_num_groups = nearest_power_of_2_divisor(out_channels, 32)
        self.short_conv = (
            eqx.nn.Sequential(
                [
                    eqx.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        use_bias=False,
                        key=key_s1,
                    ),
                    eqx.nn.GroupNorm(out_num_groups, out_channels),
                    eqx.nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=[1, 5],
                        stride=1,
                        padding=[0, 2],
                        groups=out_channels,
                        use_bias=False,
                        key=key_s2,
                    ),
                    eqx.nn.GroupNorm(out_num_groups, out_channels),
                    eqx.nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=[5, 1],
                        stride=1,
                        padding=[2, 0],
                        groups=out_channels,
                        use_bias=False,
                        key=key_s3,
                    ),
                    eqx.nn.GroupNorm(out_num_groups, out_channels),
                ]
            )
            if mode == "shortcut"
            else eqx.nn.Identity()
        )

    def training_features(self, x):
        x1 = jax.tree_util.tree_reduce(
            operator.add, [conv(x) for conv in self.primary_rpr_conv]
        )
        x1 = self.primary_activation(self.primary_shared_norm(x1))

        cheap_branches = [self.cheap_rpr_skip(x1), self.cheap_rpr_scale(x1)] + [
            conv(x1) for conv in self.cheap_rpr_conv
        ]

        x2 = jax.tree_util.tree_reduce(operator.add, cheap_branches)
        x2 = self.cheap_activation(self.cheap_shared_norm(x2))

        out = jnp.concatenate([x1, x2], axis=0)

        return out

    def inference_features(self, x):
        x1 = self.primary_activation(self.primary_shared_norm(self.primary_conv(x)))
        x2 = self.cheap_activation(self.cheap_shared_norm(self.cheap_operation(x1)))

        out = jnp.concatenate([x1, x2], axis=0)

        return out

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ):
        # key_p, key_c = jr.split(key, 2)
        # key_ps = jr.split(key_p, self.num_conv_branches)
        # key_cs = jr.split(key_c, self.num_conv_branches)

        if self.inference:
            out = self.inference_features(x)
        else:
            out = self.training_features(x)

        if self.mode == "shortcut":
            res = self.short_conv(reduce(x, "c (h 2) (w 2) -> c h w", reduction="mean"))
            gating_signal = jax.image.resize(
                image=jax.nn.sigmoid(res),
                shape=(
                    res.shape[0],
                    out.shape[1],
                    out.shape[2],
                ),  # (channels, height, width)
                method="nearest",
            )
            out = out[: self.out_channels, :, :] * gating_signal

        return out


def update_fused_ghostmodule(
    module: GenericGhostModule,
) -> GenericGhostModule:
    """
    Fuses the weights of parallel training branches into single inference convolutions.

    This function should be called after training and before switching the model
    to inference mode. It operates on a single GenericGhostModule.

    Args:
        module: An instance of GenericGhostModule with trained weights.

    Returns:
        A new GenericGhostModule with the `primary_conv` and `cheap_operation`
        weights updated for inference.
    """
    # Aggregate weights
    primary_weights = jax.tree_util.tree_reduce(
        operator.add, [conv.weight for conv in module.primary_rpr_conv]
    )

    # Aggregate biases (if they exist)
    primary_biases = None
    if module.primary_rpr_conv[0].bias is not None:
        primary_biases = jax.tree_util.tree_reduce(
            operator.add, [conv.bias for conv in module.primary_rpr_conv]
        )

    # Create the new fused primary convolution layer
    fused_primary_conv = eqx.tree_at(
        lambda conv: (conv.weight, conv.bias),
        module.primary_conv,
        (primary_weights, primary_biases),
    )

    # The cheap operation fuses the skip, scale, and depthwise-like conv branches.
    target_kernel_size = module.cheap_rpr_conv[0].weight.shape[-2:]

    def pad_kernel_to_target(kernel: Array, target_size: tuple[int, int]) -> Array:
        current_size = kernel.shape[-2:]
        if current_size == target_size:
            return kernel

        pad_h = (target_size[0] - current_size[0]) // 2
        pad_w = (target_size[1] - current_size[1]) // 2

        # JAX padding format: ((before, after), (before, after), ...)
        padding_config = [
            (0, 0),  # out_channels
            (0, 0),  # in_channels
            (pad_h, pad_h),  # height
            (pad_w, pad_w),  # width
        ]
        return jnp.pad(kernel, padding_config, "constant", constant_values=0)

    # Pad the 1x1 kernels from the scale and skip branches.
    padded_scale_weight = pad_kernel_to_target(
        module.cheap_rpr_scale.weight, target_kernel_size
    )
    padded_skip_weight = pad_kernel_to_target(
        module.cheap_rpr_skip.weight, target_kernel_size
    )

    # Sum the main cheap convolution weights.
    cheap_conv_weights = jax.tree_util.tree_reduce(
        operator.add, [conv.weight for conv in module.cheap_rpr_conv]
    )

    # Sum all weights: main convs + padded scale + padded skip.
    fused_cheap_weights = cheap_conv_weights + padded_scale_weight + padded_skip_weight

    fused_cheap_biases = None
    if module.cheap_rpr_conv[0].bias is not None:
        cheap_conv_biases = jax.tree_util.tree_reduce(
            operator.add, [conv.bias for conv in module.cheap_rpr_conv]
        )
        scale_bias = module.cheap_rpr_scale.bias
        skip_bias = module.cheap_rpr_skip.bias
        fused_cheap_biases = cheap_conv_biases + scale_bias + skip_bias

    fused_cheap_operation = eqx.tree_at(
        lambda conv: (conv.weight, conv.bias),
        module.cheap_operation,
        (fused_cheap_weights, fused_cheap_biases),
    )

    module = eqx.tree_at(lambda m: m.primary_conv, module, fused_primary_conv)
    module = eqx.tree_at(lambda m: m.cheap_operation, module, fused_cheap_operation)

    return module


def update_ghostnet(model: eqx.Module) -> eqx.Module:
    """
    Update the inference layers of a GhostNet-like model.
    Useful for intermediary evals.
    """

    def _update_leaf(module: GenericGhostModule) -> GenericGhostModule:
        if not isinstance(module, GenericGhostModule):
            return module
        return update_fused_ghostmodule(module)

    is_g_module = lambda m: isinstance(m, GenericGhostModule)
    return jax.tree_util.tree_map(_update_leaf, model, is_leaf=is_g_module)


def finalize_ghostnet(model: eqx.Module) -> eqx.Module:
    """
    Finalizes a trained GhostNet-like model for inference.

    This function recursively traverses the model and for each GenericGhostModule:
    1. Fuses the parallel convolutional branches.
    2. Replaces the training-only branches with Identity layers.
    """

    def _finalize_leaf(module: GenericGhostModule) -> GenericGhostModule:
        if not isinstance(module, GenericGhostModule):
            return module

        fused_module = update_fused_ghostmodule(module)

        # identity_list = [eqx.nn.Identity()] * len(module.primary_rpr_conv)
        final_module = eqx.tree_at(lambda m: m.primary_rpr_conv, fused_module, list())
        final_module = eqx.tree_at(lambda m: m.cheap_rpr_conv, final_module, list())
        final_module = eqx.tree_at(
            lambda m: m.cheap_rpr_scale, final_module, eqx.nn.Identity()
        )
        final_module = eqx.tree_at(
            lambda m: m.cheap_rpr_skip, final_module, eqx.nn.Identity()
        )

        return final_module

    is_g_module = lambda m: isinstance(m, GenericGhostModule)
    return jax.tree_util.tree_map(_finalize_leaf, model, is_leaf=is_g_module)


class PartialConv2d(eqx.Module):
    """Partial 2D convolution on the channel dimension.

    This layer applies a standard 2D convolution only to the first `C // n_dim`
    input channels and leaves the remaining channels untouched (identity). It
    follows the "partial convolution" idea used to increase throughput by
    reducing compute on a subset of channels while preserving overall tensor
    shape, as explored in [1].

    References:
      [1]. Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
           Networks" [arXiv:2303.03667](https://arxiv.org/abs/2303.03667).

    Implementation details:
    - Let `C` be `in_channels` and `c = C // n_dim`. Only the first `c`
      channels are convolved with a `Conv2d(c, c, ...)`. The remaining
      `C - c` channels are forwarded unchanged.
    - The forward pass uses a functional "update-slice" pattern
      (`x.at[:c, ...].set(y1)`), which compiles to an efficient
      `dynamic_update_slice` under XLA. With JIT buffer donation, this can be
      performed in-place by the compiler.
    - Spatial dimensions must be preserved by the convolution so that the
      updated slice matches the input slice shape (e.g., use `stride=1` and
      `padding="SAME"`). If spatial dimensions change, the slice update will
      fail with a shape error.

    Attributes
    - dim: Number of channels to be convolved, computed as `in_channels // n_dim`.
           This is treated as a static field for compilation stability.
    - conv: The underlying `eqx.nn.Conv2d(c, c, ...)` applied to the first `c`
            channels.

    Notes
    - FLOPs reduction is approximately `c / C = 1 / n_dim` relative to a full
      convolution with the same kernel.
    """

    dim: int = eqx.field(static=True)

    conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        n_dim: int,
        *,
        key: PRNGKeyArray,
        kernel_size: int = 3,
        padding: str | int = "SAME",
        use_bias: bool = False,
        **kwargs,
    ):
        """
        Initialize a PartialConv2d layer.

        Parameters
        - in_channels: Total number of input channels `C`.
        - n_dim: Divisor used to determine the number of convolved channels.
                 The layer will convolve `C // n_dim` channels and leave the
                 remaining channels untouched. Must be > 0, and
                 `C // n_dim` must be >= 1 for a meaningful layer.
        - key: PRNG key used to initialize the underlying convolution weights.
        - kernel_size: Convolution kernel size (passed to `eqx.nn.Conv2d`).
        - padding: Convolution padding (passed to `eqx.nn.Conv2d`). Use
                   `"SAME"` to preserve spatial dimensions with `stride=1`.
                   Integer or other forms supported by `eqx.nn.Conv2d` are also
                   accepted, but must preserve H and W for the slice update.
        - use_bias: Whether to include a bias term in the underlying convolution.
        - **kwargs: Forwarded to `eqx.nn.Conv2d` (e.g., `dilation`, `groups`).
        """
        self.dim = in_channels // n_dim
        assert self.dim >= 1, "in_channels // n_dim must be >= 1"
        assert self.dim <= in_channels, (
            "Computed convolved channels exceed total channels"
        )

        if isinstance(padding, str):
            assert padding.upper() == "SAME", (
                'When padding is a string, it must be "SAME"'
            )
        else:
            # If padding is numeric, ensure it preserves H, W for stride=1
            # For Conv2d with dilation d and kernel k, effective kernel is k_eff = (k - 1) * d + 1.
            # Output H' = H + 2*pad - k_eff + 1; preserving H requires k_eff odd and pad = k_eff // 2.
            dilation = kwargs.get("dilation", 1)
            if isinstance(dilation, (tuple, list)):
                # Require isotropic dilation for simplicity
                assert len(dilation) == 2 and dilation[0] == dilation[1], (
                    "dilation must be an int or an equal pair"
                )
                dilation = dilation[0]
            assert isinstance(dilation, int) and dilation >= 1, (
                "dilation must be a positive int"
            )
            assert isinstance(kernel_size, int) and kernel_size >= 1, (
                "kernel_size must be a positive int"
            )

            k_eff = (kernel_size - 1) * dilation + 1
            assert isinstance(padding, int) and padding >= 0, (
                "padding must be a non-negative int"
            )
            assert k_eff % 2 == 1 and padding == k_eff // 2, (
                "Integer padding must preserve spatial size: require effective kernel odd and "
                f"padding == ((kernel_size-1)*dilation+1)//2; got k_eff={k_eff}, padding={padding}"
            )

        self.conv = eqx.nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            use_bias=use_bias,
            key=key,
            **kwargs,
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        *args,
        **kwargs,
    ) -> Float[Array, "channels height width"]:
        c = self.dim
        x1 = x[:c, :, :]
        y1 = self.conv(x1)

        return x.at[:c, :, :].set(y1)


class FasterNetBlock(eqx.Module):
    """
    FasterNet-style residual block with Partial Convolution-based spatial mixing
    and pointwise MLP, adapted to Equinox/JAX.

    Structure
    - Spatial mixing: `PartialConv2d` applies a 3×3 convolution to the first
      `C // n_dim` channels and leaves the remaining channels unchanged. This
      reduces compute while keeping the tensor shape intact. See [1].
    - Channel MLP: two pointwise (1*1) convolutions expand and then project
      channels (`C -> mlp_ratio*C -> C`), with normalization and activation
      in between.
    - Regularization: includes dropout after the MLP and optional stochastic
      depth (DropPath) on the residual branch.
    - Residual: optional residual connection `y = x + DropPath(MLP(SpatialMix(x)))`.

    Shape invariants
    - Input: `[channels, height, width]`
    - Output: `[channels, height, width]` (same spatial and channel dimensions)

    References
    - [1] Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural
          Networks" [arXiv:2303.03667](https://arxiv.org/abs/2303.03667).

    Attributes
    - residual: Whether to use a residual connection with stochastic depth.
    - spatial_mixing: `PartialConv2d` performing partial 3×3 spatial mixing.
    - pw_conv1: Pointwise convolution expanding channels to `mlp_ratio * C`.
    - pw_conv2: Pointwise convolution projecting channels back to `C`.
    - norm: Normalization layer applied on the expanded channels.
    - act: Activation applied after normalization (defaults to identity if none).
    - dropout: Dropout applied after the MLP.
    - drop_path: Stochastic depth module for residual addition.
    """

    residual: bool = eqx.field(static=True)

    spatial_mixing: eqx.nn.Conv2d
    pw_conv1: eqx.nn.Conv2d
    pw_conv2: eqx.nn.Conv2d
    norm: eqx.Module
    act: eqx.Module
    dropout: eqx.nn.Dropout
    drop_path: DropPathAdd

    def __init__(
        self,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        n_dim: int = 4,
        mlp_ratio: int = 3,
        kernel_size: int = 3,
        padding: str | int = "SAME",
        norm_layer: eqx.Module | None = eqx.nn.GroupNorm,
        norm_max_group: int = 32,
        act_layer: Callable | None = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_kwargs: dict = {},
        residual: bool = True,
        **kwargs,
    ):
        """
        Initialize a FasterNetBlock.

        Parameters
        - in_channels: Number of input/output channels `C`.
        - key: PRNG key used to initialize submodules. Internally split for
          spatial mixing and pointwise convolutions.
        - n_dim: Divisor determining the fraction of channels convolved by
          `PartialConv2d`; the convolved channels are `C // n_dim`.
        - mlp_ratio: Expansion ratio for the MLP (1×1 convs): hidden size is
          `mlp_ratio * C`.
        - kernel_size: Kernel size for the spatial mixing convolution (passed
          to `PartialConv2d`).
        - padding: Padding for the spatial mixing convolution. Use `"SAME"`
          to preserve spatial dimensions.
        - norm_layer: Normalization constructor applied on the expanded
          channels. If `eqx.nn.GroupNorm`, the number of groups is chosen as
          the largest power-of-two divisor of `hidden_channels` not exceeding
          `norm_max_group`. If `None`, uses identity.
        - norm_max_group: Maximum group count when using `GroupNorm`.
        - act_layer: Callable used to construct an activation function. If
          `None`, uses identity. Passed to `eqx.nn.Lambda`.
        - dropout: Dropout probability applied after the MLP branch.
        - drop_path: Stochastic depth probability on the residual branch.
        - norm_kwargs: Extra keyword arguments forwarded to the normalization
          layer constructor.
        - residual: Whether to add the residual connection (with DropPath).
        - **kwargs: Reserved for future extensions; forwarded where applicable.

        Notes
        - The block preserves `[H, W]`. Ensure `PartialConv2d` is configured
          to preserve spatial dimensions (e.g., `"SAME"` padding).
        - When `norm_layer` is not `GroupNorm`, it should accept the channel
          count as its first argument.
        - The same input/output channel count `C` is used throughout the block.
        """

        key_sm, key_pw1, key_pw2 = jr.split(key, 3)
        self.residual = residual

        self.spatial_mixing = PartialConv2d(
            in_channels=in_channels, n_dim=n_dim, key=key_sm
        )

        hidden_channels = mlp_ratio * in_channels
        self.pw_conv1 = eqx.nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding="SAME",
            use_bias=False,
            key=key_pw1,
        )
        self.pw_conv2 = eqx.nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding="SAME",
            use_bias=False,
            key=key_pw2,
        )

        if norm_layer is not None:
            if norm_layer == eqx.nn.GroupNorm:
                num_groups = nearest_power_of_2_divisor(hidden_channels, norm_max_group)
                self.norm = eqx.nn.GroupNorm(num_groups, hidden_channels, **norm_kwargs)
            else:
                self.norm = norm_layer(hidden_channels, **norm_kwargs)
        else:
            self.norm = eqx.nn.Identity()

        self.dropout = eqx.nn.Dropout(dropout)
        self.drop_path = DropPathAdd(drop_path)
        self.act = eqx.nn.Lambda(act_layer) if act_layer else eqx.nn.Identity()

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "channels height width"]:
        key_dropout, key_droppath = jr.split(key, 2)
        x1 = self.spatial_mixing(x)
        out = self.dropout(
            self.pw_conv2(self.act(self.norm(self.pw_conv1(x1)))),
            inference=inference,
            key=key_dropout,
        )

        if self.residual:
            out = self.drop_path(x, out, inference=inference, key=key_droppath)

        return out
