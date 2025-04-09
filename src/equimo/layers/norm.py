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
        *args,
        **kwargs,
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
        gamma: Learnable scale parameters of size dim
    """

    gamma: Float[Array, "dim"]

    def __init__(self, dim: int, init_values: float):
        """Initialize LayerScale.

        Args:
            dim: Dimension of the input features
            init_values: Initial value for all scaling factors
        """
        self.gamma = jnp.repeat(init_values, dim)

    def __call__(self, x: Float[Array, "dim"]):
        """Apply layer scaling to input tensor.

        Args:
            x: Input tensor of shape (dim,)

        Returns:
            Scaled tensor of same shape as input
        """
        return x * self.gamma


class DyT(eqx.Module):
    """Dynamic Tanh layear.

    This layer implements the DyT layer introduced in the Transformer
    without Normalization paper[1].

    Attributes:
        init_values: Initial scale value (static)
        gamma: Learnable scale parameters of size dim

    References:
        [1]. Zhu, et al., Transformers without Normalization. 2025.
             https://arxiv.org/abs/2503.10622
    """

    alpha: Float[Array, "dim"]
    weight: Float[Array, "dim"]
    bias: Float[Array, "dim"]

    def __init__(self, dim: int, alpha_init_value: float = 0.5):
        """Initialize DyT.

        Args:
            dim: Dimension of the input features
            alpha_init_value: Initial value for the scaling factor
        """
        self.alpha = jnp.repeat(alpha_init_value, dim)
        self.weight = jnp.ones(dim)
        self.bias = jnp.zeros(dim)

    def __call__(
        self,
        x: Float[Array, "dim"],
        *args,
        **kwargs,
    ):
        """Apply dynamic tanh to input tensor.

        Args:
            x: Input tensor of shape (dim,)

        Returns:
            Scaled tensor of same shape as input
        """
        x = jnp.tanh(self.alpha * x)
        return x * self.weight + self.bias


def get_norm(module: str | eqx.Module) -> eqx.Module:
    """Get an `eqx.Module` from its common name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    match module:
        case "layernorm":
            return eqx.nn.LayerNorm
        case "rmsnorm":
            return eqx.nn.RMSNorm
        case "groupnorm":
            return eqx.nn.GroupNorm
        case "rmsnormgated":
            return RMSNormGated
        case "layerscale":
            return LayerScale
        case "dynamictanh":
            return DyT
        case _:
            raise ValueError(f"Got an unknown module string: {module}")
