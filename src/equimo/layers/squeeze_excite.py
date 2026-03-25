from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.utils import make_divisible, nearest_power_of_2_divisor

_SE_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_se(
    name: Optional[str] = None,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new SE modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _SE_REGISTRY:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_SE_REGISTRY[registry_name]}."
            )

        _SE_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_se(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get an SE ``eqx.Module`` class from its registered name."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _SE_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_SE_REGISTRY.keys())}"
        )

    return _SE_REGISTRY[module_lower]


@register_se()
class SEModule(eqx.Module):
    """Squeeze-and-Excite Module as defined in original SE-Nets [1] paper.

    Implements channel attention mechanism by:
    1. Squeeze: Global average pooling to capture channel-wise statistics
    2. Excitation: Two FC layers with reduction to capture channel dependencies
    3. Scale: Channel-wise multiplication with original features

    This implementation uses GroupNorm instead of the original BatchNorm for
    better stability in small batch scenarios.

    Output dtype always matches the input dtype.

    Attributes:
        fc1: First conv layer (channel reduction)
        fc2: Second conv layer (channel expansion)
        norm: GroupNorm layer or Identity if use_norm=False
        act: Gate activation function (default: sigmoid)

    Reference:
        [1]: Hu, et al., Squeeze-and-Excitation Networks. 2017.
             https://arxiv.org/abs/1709.01507
    """

    fc1: eqx.nn.Conv
    fc2: eqx.nn.Conv
    norm: eqx.Module
    act: Callable

    def __init__(
        self,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        rd_ratio: float = 1.0 / 16,
        rd_divisor: int = 8,
        use_norm: bool = False,
        norm_max_group: int = 32,
        act_layer: Callable = jax.nn.sigmoid,
        **kwargs,
    ):
        """Initialize SEModule.

        Args:
            in_channels: Number of input (and output) spatial channels.
            key: PRNG key for initialization.
            rd_ratio: Reduction ratio for the bottleneck hidden channels.
            rd_divisor: Hidden channel count is rounded to a multiple of this.
            use_norm: If True, insert GroupNorm after the first FC layer.
            norm_max_group: Maximum group count for GroupNorm.
            act_layer: Gate activation function applied to the excitation output
                (default: sigmoid).
            **kwargs: Ignored; present for drop-in use via the registry.
        """
        key_fc1, key_fc2 = jr.split(key, 2)
        rd_channels = make_divisible(
            in_channels * rd_ratio, rd_divisor, round_down_protect=False
        )
        self.fc1 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=rd_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_fc1,
        )
        num_groups = nearest_power_of_2_divisor(rd_channels, norm_max_group)
        self.norm = (
            eqx.nn.GroupNorm(num_groups, rd_channels)
            if use_norm
            else eqx.nn.Identity()
        )
        self.fc2 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=rd_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_fc2,
        )
        self.act = act_layer

    def __call__(
        self,
        x: Float[Array, "channels height width"],
    ) -> Float[Array, "channels height width"]:
        """Apply squeeze-and-excitation attention.

        Args:
            x: Input spatial feature map of shape (C, H, W).

        Returns:
            Channel-recalibrated feature map of the same shape and dtype.
        """
        dtype = x.dtype
        x_se = x.mean(axis=(1, 2), keepdims=True).astype(jnp.float32)
        x_se = jax.nn.relu(self.norm(self.fc1(x_se)))
        x_se = self.fc2(x_se)
        return x * self.act(x_se).astype(dtype)


@register_se()
class EffectiveSEModule(eqx.Module):
    """Efficient variant of Squeeze-and-Excitation Module.

    Simplifies the original SE module by:
    1. Using a single conv layer instead of two
    2. Using hard_sigmoid activation by default (faster than sigmoid)
    3. Removing the dimensionality reduction

    These modifications reduce computational cost while maintaining
    effectiveness for channel attention.

    Output dtype always matches the input dtype.

    Attributes:
        fc: Single convolution layer for channel attention.
        act: Gate activation function (default: hard_sigmoid).

    Reference:
        [1]: CenterMask: Real-Time Anchor-Free Instance Segmentation,
             https://arxiv.org/abs/1911.06667
    """

    fc: eqx.nn.Conv
    act: Callable

    def __init__(
        self,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        act_layer: Callable = jax.nn.hard_sigmoid,
        **kwargs,
    ):
        """Initialize EffectiveSEModule.

        Args:
            in_channels: Number of input (and output) spatial channels.
            key: PRNG key for initialization.
            act_layer: Gate activation function (default: hard_sigmoid).
            **kwargs: Ignored; present for drop-in use via the registry.
        """
        self.fc = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key,
        )
        self.act = act_layer

    def __call__(
        self,
        x: Float[Array, "channels height width"],
    ) -> Float[Array, "channels height width"]:
        """Apply efficient squeeze-and-excitation attention.

        Args:
            x: Input spatial feature map of shape (C, H, W).

        Returns:
            Channel-recalibrated feature map of the same shape and dtype.
        """
        dtype = x.dtype
        x_se = x.mean(axis=(1, 2), keepdims=True).astype(jnp.float32)
        x_se = self.fc(x_se)
        return x * self.act(x_se).astype(dtype)
