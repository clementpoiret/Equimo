from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from equimo.utils import nearest_power_of_2_divisor


def _haar_1d(dtype=jnp.float32) -> Tuple[Array, Array]:
    s2 = jnp.sqrt(jnp.array(2.0, dtype=dtype))
    h0 = jnp.array([1.0, 1.0], dtype=dtype) / s2
    h1 = jnp.array([-1.0, 1.0], dtype=dtype) / s2
    return h0, h1


def _haar_2d_kernels(dtype=jnp.float32) -> Tuple[Array, Array, Array, Array]:
    h0, h1 = _haar_1d(dtype)

    kLL = jnp.outer(h0, h0)  # low-low
    kLH = jnp.outer(h0, h1)  # low-high (horizontal detail)
    kHL = jnp.outer(h1, h0)  # high-low (vertical detail)
    kHH = jnp.outer(h1, h1)  # high-high (diagonal)

    return kLL, kHL, kLH, kHH


def _depthwise_conv2d_stride2(x_chw: Array, k2x2: Array) -> Array:
    """
    x_chw: (C, H, W)
    k2x2:  (2, 2) Haar kernel for one subband
    returns: (C, H/2, W/2), using SAME padding + stride 2 depthwise conv
    """
    C, _, _ = x_chw.shape

    x = x_chw[None, ...]  # (1, C, H, W) for lax.conv

    k = jnp.tile(k2x2[None, None, :, :], (C, 1, 1, 1))
    y = jax.lax.conv_general_dilated(
        lhs=x,
        rhs=k,
        window_strides=(2, 2),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=C,
    )
    return y[0]  # (C, H/2, W/2)


def haar_dwt_split(
    x_chw: Array, dtype=jnp.float32
) -> Tuple[Array, Array, Array, Array]:
    """
    Return (LL, HL, LH, HH), each (C, H/2, W/2)
    """
    kLL, kHL, kLH, kHH = _haar_2d_kernels(dtype)
    LL = _depthwise_conv2d_stride2(x_chw, kLL)
    HL = _depthwise_conv2d_stride2(x_chw, kHL)
    LH = _depthwise_conv2d_stride2(x_chw, kLH)
    HH = _depthwise_conv2d_stride2(x_chw, kHH)
    return LL, HL, LH, HH


class HWDConv(eqx.Module):
    mode: Literal["h_discard", "accurate"] = eqx.field(static=True)

    pre_norm: eqx.nn.GroupNorm
    proj: eqx.nn.Conv2d
    act: Callable
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 1,
        use_bias: bool = False,
        act: Callable = jax.nn.relu,
        mode: Literal["h_discard", "accurate"] = "accurate",
        dropout: float = 0.0,
        key: PRNGKeyArray,
    ):
        self.mode = mode

        # Channels entering the projection (and thus the pre-norm)
        channels_in = in_channels if mode == "h_discard" else 4 * in_channels

        # Choose groups so no group crosses sub-band boundaries.
        # For accurate mode: total_groups = 4 * groups_per_band, each band has groups_per_band groups.
        # For h_discard mode: total_groups = groups_per_band (single band).
        groups_per_band = nearest_power_of_2_divisor(in_channels, 32)
        total_groups = groups_per_band if mode == "h_discard" else 4 * groups_per_band

        self.pre_norm = eqx.nn.GroupNorm(channels=channels_in, groups=total_groups)

        self.proj = eqx.nn.Conv2d(
            channels_in,
            out_channels,
            kernel_size=kernel_size,
            use_bias=use_bias,
            padding="SAME",
            key=key,
        )
        self.act = act

        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self, x: Array, *, key: PRNGKeyArray, inference: bool = False
    ) -> Array:
        LL, HL, LH, HH = haar_dwt_split(x, dtype=x.dtype)  # each (C, H/2, W/2)

        if self.mode == "h_discard":
            y = LL  # (C, H/2, W/2)
        else:
            y = jnp.concatenate([LL, HL, LH, HH], axis=0)  # (4*C, H/2, W/2)

        y = self.pre_norm(y)

        y = self.proj(y)  # (out_ch, H/2, W/2)
        y = self.act(y)

        y = self.dropout(y, inference=inference, key=key)

        return y
