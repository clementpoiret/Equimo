import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.utils import nearest_power_of_2_divisor


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    """
    Helper function to compute the number of channels for the SE Modules.
    Taken from: https://github.com/pprp/timm/blob/e9aac412de82310e6905992e802b1ee4dc52b5d1/timm/layers/helpers.py#L25
    """
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SEModule(eqx.Module):
    """Squeeze-and-Excite Module as defined in original SE-Nets [1] paper.

    Implements channel attention mechanism by:
    1. Squeeze: Global average pooling to capture channel-wise statistics
    2. Excitation: Two FC layers with reduction to capture channel dependencies
    3. Scale: Channel-wise multiplication with original features

    This implementation uses GroupNorm instead of the original BatchNorm for
    better stability in small batch scenarios.

    Attributes:
        fc1: First conv layer (channel reduction)
        fc2: Second conv layer (channel expansion)
        norm: GroupNorm layer or Identity if use_norm=False

    Reference:
        [1]: Hu, et al., Squeeze-and-Excitation Networks. 2017.
             https://arxiv.org/abs/1709.01507
    """

    fc1: eqx.nn.Conv
    fc2: eqx.nn.Conv
    norm: eqx.Module

    def __init__(
        self,
        dim: int,
        *,
        key: PRNGKeyArray,
        rd_ratio: float = 1.0 / 16,
        rd_divisor: int = 8,
        use_norm: bool = False,
        norm_max_group: int = 32,
        **kwargs,
    ):
        key_fc1, key_fc2 = jr.split(key, 2)
        rd_channels = make_divisible(dim * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=rd_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_fc1,
        )
        num_groups = nearest_power_of_2_divisor(rd_channels, norm_max_group)
        self.norm = (
            eqx.nn.GroupNorm(
                num_groups,
                rd_channels,
            )
            if use_norm
            else eqx.nn.Identity()
        )
        self.fc2 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=rd_channels,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_fc2,
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
    ) -> Float[Array, "channels height width"]:
        x_se = x.mean(axis=[1, 2], keepdims=True)
        x_se = jax.nn.relu(self.norm(self.fc1(x_se)))
        x_se = self.fc2(x_se)

        return x * jax.nn.sigmoid(x_se)


class EffectiveSEModule(eqx.Module):
    """Efficient variant of Squeeze-and-Excitation Module.

    Simplifies the original SE module by:
    1. Using a single conv layer instead of two
    2. Replacing sigmoid with hard_sigmoid activation
    3. Removing the dimensionality reduction

    These modifications reduce computational cost while maintaining
    effectiveness for channel attention.

    Attributes:
        fc: Single convolution layer for channel attention

    Reference:
        [1]: CenterMask: Real-Time Anchor-Free Instance Segmentation,
             https://arxiv.org/abs/1911.06667
    """

    fc: eqx.nn.Conv

    def __init__(
        self,
        dim: int,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.fc = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
    ) -> Float[Array, "channels height width"]:
        x_se = x.mean(axis=[1, 2], keepdims=True)
        x_se = self.fc(x_se)

        return x * jax.nn.hard_sigmoid(x_se)
