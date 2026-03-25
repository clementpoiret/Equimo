from typing import Callable, Optional

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

_NORM_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_norm(
    name: Optional[str] = None,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new norm modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _NORM_REGISTRY:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_NORM_REGISTRY[registry_name]}."
            )

        _NORM_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_norm(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get a norm `eqx.Module` class from its registered name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _NORM_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_NORM_REGISTRY.keys())}"
        )

    return _NORM_REGISTRY[module_lower]


@register_norm()
class RMSNormGated(eqx.Module):
    """Root Mean Square (RMS) Normalization with optional gating.

    Implements RMS normalization with learnable scale parameters and optional
    gating mechanism. RMS norm is similar to Layer Norm but only normalizes by
    the root mean square, without centering the mean.

    Computation is promoted to float32 for numerical stability, then cast back
    to the input dtype to preserve mixed-precision training contracts.

    Attributes:
        w: Learnable scale parameter vector of size dim
        eps: Small constant for numerical stability (static)
    """

    w: Float[Array, "dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize RMSNormGated.

        Args:
            dim: Dimension of the input features
            eps: Small constant added to the variance for numerical stability
        """
        self.w = jnp.ones(dim)
        self.eps = eps

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
            Normalized tensor of same shape as input, preserving input dtype.
        """
        dtype = x.dtype

        if z is not None:
            x = x * z

        y = x.astype(jnp.float32)
        norm = y * lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + self.eps)

        # Multiply in float32 to preserve precision, then cast back to input dtype
        return (self.w * norm).astype(dtype)


@register_norm()
class LayerScale(eqx.Module):
    """Layer scaling with per-channel learnable scale.

    Supports inputs with channel-first layout. Set `axis` to the channel
    dimension (e.g., 0 for (C, H, W), 1 for (N, C, H, W)).

    The scale is cast to the input dtype before multiplication to preserve
    mixed-precision training contracts.
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
            Scaled tensor, same shape and dtype as `x`.
        """
        if x.shape[self.axis] != self.gamma.shape[0]:
            raise ValueError(
                f"Channel mismatch: x.shape[{self.axis}]={x.shape[self.axis]} "
                f"but gamma.shape[0]={self.gamma.shape[0]}"
            )

        shape = [1] * x.ndim
        shape[self.axis] = self.gamma.shape[0]
        # Cast to input dtype to avoid silent upcasting in mixed-precision training
        scale = self.gamma.reshape(shape).astype(x.dtype)

        return x * scale


@register_norm(name="dynamictanh")
class DyT(eqx.Module):
    """Dynamic Tanh layer.

    This layer implements the DyT layer introduced in the Transformer
    without Normalization paper[1].

    The normalization computation is promoted to float32 for numerical stability,
    then cast back to the input dtype to preserve mixed-precision training contracts.

    Attributes:
        alpha: Learnable per-channel scaling factor applied before tanh
        weight: Learnable per-channel output scale
        bias: Learnable per-channel output bias

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
            alpha_init_value: Initial value for the per-channel scaling factor
        """
        self.alpha = jnp.full((dim,), alpha_init_value)
        self.weight = jnp.ones(dim)
        self.bias = jnp.zeros(dim)

    def __call__(
        self,
        x: Float[Array, "dim"],
        *args,
        **kwargs,
    ) -> Float[Array, "dim"]:
        """Apply dynamic tanh to input tensor.

        Args:
            x: Input tensor of shape (dim,)

        Returns:
            Scaled tensor of same shape and dtype as input.
        """
        dtype = x.dtype
        # Promote to float32 for tanh stability, then cast output back
        x_f32 = jnp.tanh(self.alpha * x.astype(jnp.float32))
        return (x_f32 * self.weight + self.bias).astype(dtype)


@register_norm()
class RMSNorm2d(eqx.Module):
    """RMS Normalization for 2D spatial feature maps (C, H, W).

    Normalizes over the channel dimension (axis=0) at each spatial location,
    consistent with ConvNeXt-style channel normalization. Note: this differs
    from GroupNorm(groups=1) which normalizes over spatial dimensions.

    Computation is promoted to float32 for numerical stability, then cast back
    to the input dtype to preserve mixed-precision training contracts.
    """

    eps: float = eqx.field(static=True)
    weight: Optional[Float[Array, "channels"]]

    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        """
        Args:
            channels: Number of input channels (C).
            eps: Epsilon for numerical stability.
            affine: If True, learn a per-channel scale parameter (weight).
        """
        self.eps = eps
        self.weight = jnp.ones(channels) if affine else None

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "channels height width"]:
        """
        Forward pass for a single sample (C, H, W).
        Use jax.vmap(model)(batch) for (N, C, H, W) inputs.
        """
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        # Normalize over channel dimension: result shape (1, H, W)
        var = jnp.mean(jnp.square(x_f32), axis=0, keepdims=True)
        x_norm = x_f32 * lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight[:, None, None]
        return x_norm.astype(dtype)


@register_norm()
class LayerNorm2d(eqx.Module):
    """Layer Normalization for 2D spatial feature maps (C, H, W).

    Normalizes over the channel dimension (axis=0) at each spatial location,
    consistent with ConvNeXt-style channel normalization.

    WARNING: THIS IS NOT LIKE GroupNorm(groups=1, ...) as some papers describe!
    GroupNorm(groups=1) normalizes over spatial+channel dimensions, while this
    layer normalizes over channels only at each spatial location.

    Computation is promoted to float32 for numerical stability, then cast back
    to the input dtype to preserve mixed-precision training contracts.
    """

    eps: float = eqx.field(static=True)
    weight: Optional[Float[Array, "channels"]]
    bias: Optional[Float[Array, "channels"]]

    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        self.eps = eps
        if affine:
            self.weight = jnp.ones(channels)
            self.bias = jnp.zeros(channels)
        else:
            self.weight = None
            self.bias = None

    def __call__(
        self, x: Float[Array, "channels height width"]
    ) -> Float[Array, "channels height width"]:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        # Normalize over channel dimension: mean/var shape (1, H, W)
        mean = jnp.mean(x_f32, axis=0, keepdims=True)
        var = jnp.mean(jnp.square(x_f32 - mean), axis=0, keepdims=True)
        x_norm = (x_f32 - mean) * lax.rsqrt(var + self.eps)
        if self.weight is not None and self.bias is not None:
            x_norm = x_norm * self.weight[:, None, None] + self.bias[:, None, None]
        return x_norm.astype(dtype)


# Register built-in equinox norms — these cannot be decorated, so we insert directly.
# Collision checking is skipped here; these are canonical names owned by this module.
_NORM_REGISTRY["layernorm"] = eqx.nn.LayerNorm
_NORM_REGISTRY["rmsnorm"] = eqx.nn.RMSNorm
_NORM_REGISTRY["groupnorm"] = eqx.nn.GroupNorm
