from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from equimo.utils import nearest_power_of_2_divisor

_WAVELET_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_wavelet(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new wavelet modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.

    Args:
        name: Registry key. Defaults to the lowercase class name.
        force: If True, allow overwriting an existing entry. Default False.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _WAVELET_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_WAVELET_REGISTRY[registry_name]}."
            )

        _WAVELET_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_wavelet(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get a wavelet `eqx.Module` class from its registered name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _WAVELET_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_WAVELET_REGISTRY.keys())}"
        )

    return _WAVELET_REGISTRY[module_lower]


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


@register_wavelet()
class HWDConv(eqx.Module):
    mode: Literal["h_discard", "band_grouped", "accurate"] = eqx.field(static=True)

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
        stride: int = 1,
        padding: str | int = "SAME",
        use_bias: bool = False,
        act_layer: Callable = jax.nn.relu,
        mode: Literal["h_discard", "band_grouped", "accurate"] = "accurate",
        dropout: float = 0.0,
        key: PRNGKeyArray,
    ):
        self.mode = mode

        channels_in = in_channels if mode == "h_discard" else 4 * in_channels

        # GroupNorm: align groups to sub-band boundaries
        # groups_per_band divides in_channels; total_groups = 4 * groups_per_band for 4 bands
        groups_per_band = nearest_power_of_2_divisor(in_channels, 32)
        total_groups = groups_per_band if mode == "h_discard" else 4 * groups_per_band
        self.pre_norm = eqx.nn.GroupNorm(channels=channels_in, groups=total_groups)

        # Projection 1x1 conv
        # - accurate: dense projection (groups=1)
        # - band_grouped: independent per-band projections (groups=4), no cross-band mixing
        # - h_discard: single-band dense projection (groups=1)
        proj_groups = 1
        if mode == "band_grouped":
            if out_channels % 4 != 0:
                raise ValueError(
                    "band_grouped mode requires out_channels to be divisible by 4 "
                    "so each sub-band produces an equal share."
                )
            proj_groups = 4

        self.proj = eqx.nn.Conv2d(
            in_channels=channels_in,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            groups=proj_groups,
            key=key,
        )

        self.act = act_layer

        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self, x: Array, *, key: PRNGKeyArray, inference: bool = False
    ) -> Array:
        LL, HL, LH, HH = haar_dwt_split(x, dtype=x.dtype)  # each (C, H/2, W/2)

        if self.mode == "h_discard":
            y = LL  # (C, H/2, W/2)
        else:
            # Keep sub-bands contiguous and ordered so GN groups and grouped conv line up
            y = jnp.concatenate([LL, HL, LH, HH], axis=0)  # (4*C, H/2, W/2)

        # Band-aligned normalization (no cross-band statistics)
        y = self.pre_norm(y)

        y = self.proj(y)  # (out_channels, H/2, W/2)
        y = self.act(y)

        y = self.dropout(y, inference=inference, key=key)

        return y
