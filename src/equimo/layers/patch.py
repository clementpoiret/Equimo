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

_PATCH_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_patch(
    name: Optional[str] = None,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new patch embedding/merging modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _PATCH_REGISTRY:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_PATCH_REGISTRY[registry_name]}."
            )

        _PATCH_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_patch(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get a patch `eqx.Module` class from its registered name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _PATCH_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_PATCH_REGISTRY.keys())}"
        )

    return _PATCH_REGISTRY[module_lower]


@register_patch()
class PatchEmbedding(eqx.Module):
    """Image to patch embedding module for vision transformers.

    Converts an image into a sequence of patch embeddings by:
    1. Splitting the image into fixed-size patches via a strided convolution
    2. Optionally normalizing each patch embedding
    3. Optionally flattening the spatial dimensions into a token sequence

    Supports dynamic image sizes and padding when needed. Output dtype
    always matches the input dtype.

    Attributes:
        patch_size: Size of each patch (static)
        img_size: Input image size (static)
        grid_size: Grid dimensions after patching (static)
        num_patches: Total number of patches (static)
        flatten: Whether to flatten spatial dimensions (static)
        dynamic_img_size: Allow variable image sizes (static)
        dynamic_img_pad: Allow padding for non-divisible sizes (static)
        proj: Patch projection convolution
        norm: Normalization layer applied per patch token
    """

    patch_size: int | Tuple[int, int] = eqx.field(static=True)

    img_size: Optional[Tuple[int, int]] = eqx.field(static=True)
    grid_size: Optional[Tuple[int, int]] = eqx.field(static=True)
    num_patches: Optional[int] = eqx.field(static=True)

    flatten: bool = eqx.field(static=True)
    dynamic_img_size: bool = eqx.field(static=True)
    dynamic_img_pad: bool = eqx.field(static=True)

    proj: eqx.nn.Conv
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
        norm_layer: Optional[Callable] = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        """Initialize PatchEmbedding.

        Args:
            in_channels: Number of input image channels (e.g. 3 for RGB).
            dim: Output patch embedding dimension (token latent width).
            patch_size: Size of each patch; scalar or (H, W) tuple.
            key: PRNG key for initialization.
            img_size: Expected input image size for shape validation.
                      None disables validation.
            flatten: If True, output shape is (num_patches, dim).
                     If False, output shape is (dim, grid_h, grid_w).
            dynamic_img_size: Allow images of different sizes than img_size.
            dynamic_img_pad: Pad images whose sides are not divisible by patch_size.
            norm_layer: Optional norm class called as norm_layer(dim, eps=eps).
                        Defaults to no normalization.
            eps: Epsilon passed to norm_layer.
        """
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
            kernel_size=patch_size,
            stride=patch_size,
            key=key,
        )
        self.norm = (
            norm_layer(dim, eps=eps) if norm_layer is not None else eqx.nn.Identity()
        )

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


@register_patch()
class ConvPatchEmbed(eqx.Module):
    """Convolutional Patch Embedding, used in MambaVision.

    Applies two successive stride-2 convolution stages to produce patch
    embeddings, reducing spatial resolution by 4× in each dimension.
    Each stage follows the order: Conv → Norm → Activation.

    Output dtype always matches the input dtype.

    Attributes:
        conv1: First 3×3 strided convolution (stride 2).
        conv2: Second 3×3 strided convolution (stride 2).
        norm1: Normalization applied after conv1 (in token space).
        norm2: Normalization applied after conv2 (in token space).
        act: Activation function applied after each norm.
    """

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    norm1: eqx.Module
    norm2: eqx.Module
    act: Callable = eqx.field(static=True)

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
        **kwargs,
    ):
        """Initialize ConvPatchEmbed.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of intermediate channels after conv1.
            out_channels: Number of output channels after conv2.
            key: PRNG key for initialization.
            act_layer: Activation function applied after each norm.
            norm_layer: Norm class called as norm_layer(channels, eps=eps).
            eps: Epsilon passed to norm_layer.
        """
        key_conv1, key_conv2 = jr.split(key, 2)

        self.act = act_layer

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

    def _to_tokens(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "space channels"]:
        return rearrange(x, "c h w -> (h w) c")

    def _from_tokens(
        self,
        x: Float[Array, "space channels"],
        h: int,
        w: int,
    ) -> Float[Array, "channels height width"]:
        return rearrange(x, "(h w) c -> c h w", h=h, w=w)

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "out_channels new_height new_width"]:
        """Apply two conv-norm-act stages, each halving spatial resolution.

        Args:
            x: Input tensor of shape (channels, height, width).

        Returns:
            Output tensor of shape (out_channels, height // 4, width // 4),
            same dtype as input.
        """
        c, h, w = x.shape

        x = self._from_tokens(
            self.act(jax.vmap(self.norm1)(self._to_tokens(self.conv1(x)))),
            h=h // 2,
            w=w // 2,
        )
        x = self._from_tokens(
            self.act(jax.vmap(self.norm2)(self._to_tokens(self.conv2(x)))),
            h=h // 4,
            w=w // 4,
        )
        return x


@register_patch()
class PatchMerging(eqx.Module):
    """Patch merging module that reduces spatial resolution while increasing channels.

    Implements hierarchical feature aggregation using three convolution stages:
    1. 1×1 conv to expand channels
    2. 3×3 depthwise conv with stride 2 for spatial reduction
    3. 1×1 conv to set the final channel dimension

    The module follows an inverted bottleneck architecture. By default the
    output channel count is doubled (``out_dim = 2 * dim``), matching the
    standard hierarchical ViT convention where each stage doubles width.

    Attributes:
        conv1: Channel expansion convolution (1×1).
        conv2: Spatial reduction depthwise convolution (3×3, stride 2).
        conv3: Channel projection convolution (1×1).
    """

    conv1: SingleConvBlock
    conv2: SingleConvBlock
    conv3: SingleConvBlock

    def __init__(
        self,
        dim: int,
        *,
        key: PRNGKeyArray,
        out_dim: Optional[int] = None,
        ratio: float = 4.0,
        **kwargs,
    ):
        """Initialize PatchMerging.

        Args:
            dim: Input token embedding dimension.
            key: PRNG key for initialization.
            out_dim: Output token embedding dimension. Defaults to 2 * dim,
                     following the standard hierarchical doubling convention.
            ratio: Expansion ratio for the hidden (intermediate) dimension.
                   hidden_dim = out_dim * ratio.
        """
        key_conv1, key_conv2, key_conv3 = jr.split(key, 3)

        out_dim = out_dim or int(dim * 2)
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

    def __call__(
        self, x: Float[Array, "seqlen dim"]
    ) -> Float[Array, "new_seqlen out_dim"]:
        """Apply spatial downsampling on a token sequence.

        The sequence length must represent a square spatial grid
        (i.e. seqlen must be a perfect square).

        Args:
            x: Token sequence of shape (seqlen, dim).

        Returns:
            Downsampled token sequence of shape (seqlen // 4, out_dim),
            same dtype as input.
        """
        l, _ = x.shape
        h = w = int(l**0.5)

        x = rearrange(x, "(h w) c -> c h w", h=h, w=w)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = rearrange(x, "c h w -> (h w) c")

        return x


@register_patch()
class SEPatchMerging(eqx.Module):
    """Squeeze-and-Excite Patch Merging module.

    Combines patch merging with squeeze-and-excitation channel attention.
    Architecture per stage: 1×1 expand → 3×3 reduce (stride 2) → SE → 1×1 project.
    Each convolution is followed by group normalization and an activation.

    Output dtype always matches the input dtype.

    Attributes:
        conv1: Channel expansion 1×1 convolution.
        conv2: Spatial reduction 3×3 convolution (stride 2).
        conv3: Channel projection 1×1 convolution.
        norm1: Group normalization after conv1.
        norm2: Group normalization after conv2.
        norm3: Group normalization after conv3.
        se: Squeeze-and-Excitation module.
        act: Activation function.
    """

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    conv3: eqx.nn.Conv
    norm1: eqx.Module
    norm2: eqx.Module
    norm3: eqx.Module
    se: SEModule
    act: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        expansion_ratio: float = 4.0,
        act_layer: Callable = jax.nn.relu,
        norm_max_group: int = 32,
        **kwargs,
    ):
        """Initialize SEPatchMerging.

        Args:
            in_channels: Number of input spatial channels.
            out_channels: Number of output spatial channels.
            key: PRNG key for initialization.
            expansion_ratio: Channel expansion factor for the hidden dimension.
                             hidden_channels = in_channels * expansion_ratio.
            act_layer: Activation function applied after each norm.
            norm_max_group: Maximum number of groups for GroupNorm; actual
                            group count is the largest power-of-2 divisor of
                            the channel count up to this value.
        """
        key_conv1, key_conv2, key_conv3, key_se = jr.split(key, 4)

        self.act = act_layer

        hidden_channels = int(in_channels * expansion_ratio)
        num_groups_hidden = nearest_power_of_2_divisor(hidden_channels, norm_max_group)

        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv1,
        )
        self.norm1 = eqx.nn.GroupNorm(num_groups_hidden, hidden_channels)
        self.conv2 = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            key=key_conv2,
        )
        self.norm2 = eqx.nn.GroupNorm(num_groups_hidden, hidden_channels)
        self.se = SEModule(hidden_channels, rd_ratio=1.0 / 4, key=key_se)
        self.conv3 = eqx.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv3,
        )
        self.norm3 = eqx.nn.GroupNorm(
            nearest_power_of_2_divisor(out_channels, norm_max_group),
            out_channels,
        )

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "out_channels new_height new_width"]:
        """Apply SE-augmented spatial downsampling.

        Args:
            x: Input tensor of shape (channels, height, width).

        Returns:
            Output tensor of shape (out_channels, height // 2, width // 2),
            same dtype as input.
        """
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.se(x)
        x = self.norm3(self.conv3(x))
        return x
