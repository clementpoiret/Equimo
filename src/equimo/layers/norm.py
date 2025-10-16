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
    """Layer scaling with per-channel learnable scale.

    Supports inputs with channel-first layout. Set `axis` to the channel
    dimension (e.g., 0 for (C, H, W), 1 for (N, C, H, W)).
    """

    gamma: Float[Array, "C"]
    axis: int = eqx.field(static=True)

    def __init__(
        self, dim: int, init_values: float = 1e-6, axis: int = 0, dtype=jnp.float32
    ):
        """Initialize LayerScale.

        Args:
            dim: Number of channels (size of the channel dimension).
            init_values: Initial scale value for all channels (typically small, e.g., 1e-6).
            axis: Index of the channel dimension in the input.
            dtype: Data type for the scale parameters.
        """
        self.gamma = jnp.full((dim,), init_values, dtype=dtype)
        self.axis = axis

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Apply per-channel scaling.

        Args:
            x: Input tensor. The size along `axis` must equal `dim`.

        Returns:
            Scaled tensor, same shape as `x`.
        """
        # Validate channel dimension
        if x.shape[self.axis] != self.gamma.shape[0]:
            raise ValueError(
                f"Channel mismatch: x.shape[{self.axis}]={x.shape[self.axis]} "
                f"but gamma.shape[0]={self.gamma.shape[0]}"
            )

        # Broadcast gamma across non-channel dimensions
        shape = [1] * x.ndim
        shape[self.axis] = self.gamma.shape[0]
        scale = self.gamma.reshape(shape)

        return x * scale


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
