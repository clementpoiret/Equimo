from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float


class RMSNormGated(eqx.Module):
    """Root Mean Square (RMS) Normalization with optional gating.

    Implements RMS normalization with learnable scale parameters and optional
    gating mechanism. RMS norm is similar to Layer Norm but only normalizes by
    the root mean square, without centering the mean.

    Attributes:
        w: Learnable scale parameter vector of size dim
    """

    w: Float[Array, "dim"]

    def __init__(self, d: int):
        """Initialize RMSNormGated.

        Args:
            d: Dimension of the input features
        """
        self.w = jnp.ones(d)

    def __call__(
        self,
        x: Float[Array, "dim"],
        z: Optional[Float[Array, "dim"]] = None,
    ) -> Float[Array, "dim"]:
        """Apply RMS normalization with optional gating.

        Args:
            x: Input tensor of shape (dim,)
            z: Optional gating tensor of shape (dim,)

        Returns:
            Normalized tensor of same shape as input
        """
        if z is not None:
            x *= z

        y = x.astype(jnp.float32)
        norm = y * lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + 1e-5)

        return self.w * norm.astype(x.dtype)


class LayerScale(eqx.Module):
    """Layer scaling module for stabilizing deep networks.

    Implements learnable scaling factors for each feature dimension, initialized
    to a small value. This helps stabilize training in deep networks by initially
    dampening the contribution of each layer.

    Attributes:
        init_values: Initial scale value (static)
        gamma: Learnable scale parameters of size dim
    """

    init_values: float = eqx.field(static=True)
    gamma: Optional[Float[Array, "dim"]]

    def __init__(self, dim: int, init_values: float):
        """Initialize LayerScale.

        Args:
            dim: Dimension of the input features
            init_values: Initial value for all scaling factors
        """
        self.init_values = init_values
        self.gamma = jnp.repeat(self.init_values, dim)

    def __call__(self, x: Float[Array, "dim"]):
        """Apply layer scaling to input tensor.

        Args:
            x: Input tensor of shape (dim,)

        Returns:
            Scaled tensor of same shape as input
        """
        return x * self.gamma
