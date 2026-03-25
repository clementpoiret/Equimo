from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.convolution import DoubleConvBlock, SingleConvBlock
from equimo.layers.generic import Residual
from equimo.layers.patch import SEPatchMerging
from equimo.utils import nearest_power_of_2_divisor

_DOWNSAMPLER_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_downsampler(
    name: Optional[str] = None,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new downsampler modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _DOWNSAMPLER_REGISTRY:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_DOWNSAMPLER_REGISTRY[registry_name]}."
            )

        _DOWNSAMPLER_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_downsampler(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get a downsampler ``eqx.Module`` class from its registered name."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _DOWNSAMPLER_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_DOWNSAMPLER_REGISTRY.keys())}"
        )

    return _DOWNSAMPLER_REGISTRY[module_lower]


@register_downsampler()
class ConvNormDownsampler(eqx.Module):
    """A module that performs spatial downsampling using strided convolution.

    This module reduces spatial dimensions (height and width) by a factor of 2
    (``"simple"`` mode) or 4 (``"double"`` mode) while optionally increasing the
    channel dimension.

    Modes:
        ``"simple"``: A single 3×3 strided conv followed by optional GroupNorm.
            Spatial dimensions are halved (stride 2).
        ``"double"``: Two consecutive 3×3 strided convs each with stride 2,
            separated by optional GroupNorm and an activation. Spatial dimensions
            are quartered overall. The intermediate channel count is
            ``out_channels // 2``.

    Note:
        ``act_layer`` is only applied in ``"double"`` mode between the two
        convolution layers. It is ignored in ``"simple"`` mode.

    Attributes:
        downsampler: Sequential module that performs the downsampling.
    """

    downsampler: eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int | None = None,
        act_layer: Callable | None = None,
        use_bias: bool = False,
        use_norm: bool = True,
        mode: Literal["double", "simple"] = "simple",
        key: PRNGKeyArray,
    ):
        """Initialize ConvNormDownsampler.

        Args:
            in_channels: Number of input spatial channels.
            out_channels: Number of output spatial channels (default: 2 × in_channels).
            act_layer: Activation function inserted between the two convolutions in
                ``"double"`` mode. Ignored in ``"simple"`` mode. Default: None.
            use_bias: Whether to use bias in the convolutional layers.
            use_norm: Whether to insert GroupNorm after each convolution.
            mode: Downsampling strategy — ``"simple"`` (2×) or ``"double"`` (4×).
            key: PRNG key for initialization.
        """
        out_channels = out_channels if out_channels else 2 * in_channels

        match mode:
            case "simple":
                layers = [
                    eqx.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        use_bias=use_bias,
                        key=key,
                    ),
                    eqx.nn.GroupNorm(
                        nearest_power_of_2_divisor(out_channels, 32), out_channels
                    )
                    if use_norm
                    else eqx.nn.Identity(),
                ]
                if act_layer is not None:
                    layers.append(eqx.nn.Lambda(act_layer))
                self.downsampler = eqx.nn.Sequential(layers)

            case "double":
                _d = out_channels // 2
                layers = [
                    eqx.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=_d,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        use_bias=use_bias,
                        key=jr.fold_in(key, 0),
                    ),
                    eqx.nn.GroupNorm(nearest_power_of_2_divisor(_d, 32), _d)
                    if use_norm
                    else eqx.nn.Identity(),
                ]
                if act_layer is not None:
                    layers.append(eqx.nn.Lambda(act_layer))
                layers += [
                    eqx.nn.Conv2d(
                        in_channels=_d,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        use_bias=use_bias,
                        key=jr.fold_in(key, 1),
                    ),
                    eqx.nn.GroupNorm(
                        nearest_power_of_2_divisor(out_channels, 32), out_channels
                    )
                    if use_norm
                    else eqx.nn.Identity(),
                ]
                self.downsampler = eqx.nn.Sequential(layers)

    def __call__(
        self,
        x: Float[Array, "in_channels height width"],
        *args,
        **kwargs,
    ) -> Float[Array, "out_channels new_height new_width"]:
        return self.downsampler(x)


@register_downsampler()
class PWSEDownsampler(eqx.Module):
    """Downsampling module for spatial feature reduction.

    Combines convolution blocks and SE patch merging to reduce spatial dimensions
    (by 2×) while increasing the channel count. Uses residual connections and
    alternates between depthwise and pointwise convolutions.

    The pipeline is:
    1. Depthwise 3×3 conv with residual (``conv1``)
    2. Pointwise 1×1 double conv (``conv2``)
    3. SE patch merging — halves spatial dims, changes channel count (``patch_merging``)
    4. Depthwise 3×3 conv with residual (``conv3``)
    5. Pointwise 1×1 double conv (``conv4``)

    Attributes:
        conv1: Depthwise convolution with residual connection.
        conv2: Pointwise double convolution block.
        conv3: Depthwise convolution with residual connection (post-merge).
        conv4: Pointwise double convolution block (post-merge).
        patch_merging: SE patch merging layer that halves spatial dimensions.
    """

    conv1: eqx.Module
    conv2: eqx.Module
    conv3: eqx.Module
    conv4: eqx.Module
    patch_merging: eqx.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        drop_path: float = 0.0,
        **kwargs,
    ):
        """Initialize PWSEDownsampler.

        Args:
            in_channels: Number of input spatial channels.
            out_channels: Number of output spatial channels.
            key: PRNG key for initialization.
            drop_path: Drop path rate applied inside conv blocks.
            **kwargs: Ignored; present for drop-in use via the registry.
        """
        key_conv1, key_conv2, key_conv3, key_conv4, key_pm = jr.split(key, 5)
        self.conv1 = Residual(
            SingleConvBlock(
                in_channels,
                in_channels,
                act_layer=None,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                key=key_conv1,
            ),
            drop_path=drop_path,
        )
        self.conv2 = DoubleConvBlock(
            in_channels=in_channels,
            hidden_channels=in_channels * 2,
            act_layer=None,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv2,
        )
        self.patch_merging = SEPatchMerging(
            in_channels=in_channels,
            out_channels=out_channels,
            key=key_pm,
        )
        self.conv3 = Residual(
            SingleConvBlock(
                out_channels,
                out_channels,
                act_layer=None,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_channels,
                key=key_conv3,
            ),
            drop_path=drop_path,
        )
        self.conv4 = DoubleConvBlock(
            in_channels=out_channels,
            hidden_channels=out_channels * 2,
            act_layer=None,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv4,
        )

    def __call__(
        self,
        x: Float[Array, "in_channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "out_channels new_height new_width"]:
        """Apply downsampling to input features.

        Args:
            x: Input spatial feature tensor of shape (C, H, W).
            key: PRNG key for stochastic operations (drop path).
            inference: If True, disables stochastic operations.

        Returns:
            Downsampled feature tensor with halved spatial dimensions and
            increased channel count.
        """
        key_conv1, key_conv2, key_conv3, key_conv4 = jr.split(key, 4)
        x = self.conv2(
            self.conv1(x, inference=inference, key=key_conv1),
            inference=inference,
            key=key_conv2,
        )
        x = self.patch_merging(x)
        x = self.conv4(
            self.conv3(x, inference=inference, key=key_conv3),
            inference=inference,
            key=key_conv4,
        )

        return x
