from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.convolution import ConvBlock, SingleConvBlock
from equimo.layers.generic import Residual
from equimo.layers.patch import SEPatchMerging
from equimo.utils import nearest_power_of_2_divisor


class ConvNormDownsampler(eqx.Module):
    """A module that performs spatial downsampling using strided convolution.

    This module reduces spatial dimensions (height and width) by a factor of 2
    while optionally increasing the channel dimension. Uses a 3x3 strided
    convolution for downsampling.

    Attributes:
        reduction: Convolutional layer that performs the downsampling
    """

    downsampler: eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int | None = None,
        act_layer: Callable = jax.nn.gelu,
        use_bias: bool = False,
        use_norm: bool = True,
        mode: Literal["double", "simple"] = "simple",
        key: PRNGKeyArray,
    ):
        out_channels = out_channels if out_channels else 2 * in_channels

        match mode:
            case "simple":
                self.downsampler = eqx.nn.Sequential(
                    [
                        eqx.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            use_bias=False,
                            key=key,
                        ),
                        eqx.nn.GroupNorm(
                            nearest_power_of_2_divisor(out_channels, 32), out_channels
                        )
                        if use_norm
                        else eqx.nn.Identity(),
                    ]
                )
            case "double":
                self.downsampler = eqx.nn.Sequential(
                    [
                        eqx.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=(_d := out_channels // 2),
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            use_bias=use_bias,
                            key=jr.fold_in(key, 0),
                        ),
                        eqx.nn.GroupNorm(nearest_power_of_2_divisor(_d, 32), _d)
                        if use_norm
                        else eqx.nn.Identity(),
                        eqx.nn.Lambda(act_layer),
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
                )

    def __call__(self, x, *args, **kwargs):
        return self.downsampler(x)


class PWSEDownsampler(eqx.Module):
    """Downsampling module for spatial feature reduction.

    Combines convolution blocks and patch merging to reduce spatial dimensions
    while increasing feature channels. Uses residual connections and alternates
    between depthwise and pointwise convolutions.

    Attributes:
        conv1: First depthwise convolution with residual
        conv2: First pointwise convolution block
        conv3: Second depthwise convolution with residual
        conv4: Second pointwise convolution block
        patch_merging: Squeeze-and-excitation patch merging layer
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
        self.conv2 = ConvBlock(
            in_channels,
            hidden_in_channels=in_channels * 2,
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
        self.conv4 = ConvBlock(
            out_channels,
            hidden_in_channels=out_channels * 2,
            act_layer=None,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv4,
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        """Apply downsampling to input features.

        Args:
            x: Input feature tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Downsampled feature tensor with increased channels
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
