from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


class WeightNormLinear(eqx.Module):
    """Linear layer with weight normalization.

    Implements weight normalization as described in "Weight Normalization: A Simple
    Reparameterization to Accelerate Training of Deep Neural Networks". The weights
    are parameterized as w = g * (v/||v||), where g is a scalar and v is a vector.

    Attributes:
        weight_v: The weight vector v to be normalized
        weight_g: The scalar multiplier g for controlling the output scale

    References:
        [1] https://arxiv.org/abs/1602.07868
    """

    weight_v: jnp.ndarray
    weight_g: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, key: PRNGKeyArray):
        self.weight_v = eqx.nn.Linear(
            in_features, out_features, use_bias=False, key=key
        ).weight
        self.weight_g = jnp.ones((out_features, 1))

    def __call__(self, x):
        """Apply weight normalized linear transformation.

        Args:
            x: Input tensor

        Returns:
            Transformed tensor using normalized weights
        """
        v_norm = jnp.linalg.norm(self.weight_v, ord=2, axis=1, keepdims=True)
        normalized_v = self.weight_v / v_norm
        weight = self.weight_g * normalized_v

        out = x @ weight.T
        return out


class DINOHead(eqx.Module):
    """Multi-layer Perceptron (MLP) head used to train DINOv2 models.

    Implements the projection head architecture from DINOv2 self-supervised learning.
    Features multiple fully connected layers with activation, followed by L2
    normalization and a weight-normalized final layer.

    The architecture follows:
    input -> fc1 -> act -> fc2 -> act -> fc3 -> act -> L2norm -> weight_norm_linear

    Attributes:
        fc1: First linear layer projecting to hidden dimension
        fc2: Second linear layer maintaining hidden dimension
        fc3: Third linear layer projecting to bottleneck dimension
        last: Final weight-normalized linear layer
        act_layer: Activation function used between layers

    References:
        [1] https://arxiv.org/abs/2304.07193
    """

    act_layer: Callable = eqx.field(static=True)

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    last: WeightNormLinear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_dim=2048,
        bottleneck_dim=256,
        key: PRNGKeyArray,
        act_layer: Callable = jax.nn.gelu,
        **kwargs,
    ):
        """Initialize the DINOv2 projection head.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            hidden_dim: Dimension of hidden layers (default: 2048)
            bottleneck_dim: Dimension of bottleneck layer (default: 256)
            key: PRNG key for initialization
            act_layer: Activation function (default: gelu)
            **kwargs: Additional arguments
        """
        key_fc1, key_fc2, key_fc3, key_last = jr.split(key, 4)

        self.act_layer = act_layer

        self.fc1 = eqx.nn.Linear(in_features, hidden_dim, key=key_fc1)
        self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=key_fc2)
        self.fc3 = eqx.nn.Linear(hidden_dim, bottleneck_dim, key=key_fc3)
        self.last = WeightNormLinear(bottleneck_dim, out_features, key=key_last)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        """Process input through the DINOv2 projection head.

        Args:
            x: Input feature tensor
            inference: Whether to enable dropout (unused in original implementation)
            key: PRNG key for random operations

        Returns:
            Projected and normalized features
        """
        eps = 1e-6 if x.dtype == jnp.float16 else 1e-12
        x = self.act_layer(jax.vmap(self.fc1)(x))
        x = self.act_layer(jax.vmap(self.fc2)(x))
        x = self.act_layer(jax.vmap(self.fc3)(x))

        x = x / (jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True) + eps)

        x = self.last(x)

        return x


class Mlp(eqx.Module):
    """Multi-layer Perceptron (MLP) module with dropout.

    A standard MLP implementation with two fully connected layers, activation function,
    and dropout for regularization. The architecture follows:
    input -> fc1 -> activation -> dropout1 -> fc2 -> dropout2 -> output

    Attributes:
        fc1: First linear layer
        fc2: Second linear layer
        norm: Optional norm between fc1 and fc2
        drop1: Dropout after first layer
        drop2: Dropout after second layer
        act_layer: Activation function
    """

    act_layer: Callable = eqx.field(static=True)

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    norm: eqx.Module
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout

    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        out_features: int | None = None,
        hidden_features: int | None = None,
        act_layer: Callable = jax.nn.gelu,
        norm_layer: Callable | None = None,
        dropout_rate: float = 0.0,
        bias: bool = True,
        eps: float = 1e-5,
        **kwargs,
    ):
        """Initialize the MLP.

        Args:
            in_features: Number of input features
            key: PRNG key for initialization
            out_features: Number of output features (default: same as in_features)
            hidden_features: Number of hidden features (default: same as in_features)
            act_layer: Activation function (default: gelu)
            norm_layer: Optional norm layer to apply between denses (default: None)
            dropout_rate: Dropout probability (default: 0.0)
            bias: Whether to include bias in linear layers (default: True)
            **kwargs: Additional arguments
        """
        key_fc1, key_fc2 = jr.split(key, 2)

        self.act_layer = act_layer

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = eqx.nn.Linear(
            in_features, hidden_features, use_bias=bias, key=key_fc1
        )
        self.norm = (
            norm_layer(hidden_features, eps=eps) if norm_layer else eqx.nn.Identity()
        )
        self.fc2 = eqx.nn.Linear(
            hidden_features, out_features, use_bias=bias, key=key_fc2
        )

        self.drop1 = eqx.nn.Dropout(dropout_rate)
        self.drop2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        mask: Optional[Float[Array, ""]] = None,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key_dr1, key_dr2 = jr.split(key, 2)

        x = self.drop1(
            jax.vmap(self.norm)(self.act_layer(jax.vmap(self.fc1)(x))),
            inference=inference,
            key=key_dr1,
        )

        if mask is not None:
            x *= mask

        x = self.drop2(
            jax.vmap(self.fc2)(x),
            inference=inference,
            key=key_dr2,
        )

        if mask is not None:
            x *= mask

        return x


class SwiGlu(eqx.Module):
    """SwiGLU activation module with dropout.

    Implements the SwiGLU (Swish-Gated Linear Unit) activation function with dropout,
    as described in "GLU Variants Improve Transformer" paper [1]. The architecture uses
    a gating mechanism where the input is transformed by two parallel paths and
    combined multiplicatively.

    The computation flow is:
    1. Joint projection to higher dimension (w12)
    2. Split into two paths
    3. Apply SiLU to first path and multiply with second path
    4. Project back to original dimension (w3)

    Attributes:
        w12: Joint projection layer for both paths
        w3: Final projection layer
        drop1: Dropout after gating
        drop2: Dropout after final projection

    References:
        [1]: https://arxiv.org/pdf/2002.05202
    """

    w12: eqx.nn.Linear
    w3: eqx.nn.Linear
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout

    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        out_features: int | None = None,
        hidden_features: int | None = None,
        dropout_rate: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        """Initialize the SwiGLU module.

        Args:
            in_features: Number of input features
            key: PRNG key for initialization
            out_features: Number of output features (default: same as in_features)
            hidden_features: Size of hidden dimension (default: same as in_features)
            dropout_rate: Dropout probability (default: 0.0)
            bias: Whether to include bias in linear layers (default: True)
            **kwargs: Additional arguments
        """
        key_fc1, key_fc2 = jr.split(key, 2)

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.w12 = eqx.nn.Linear(
            in_features, 2 * hidden_features, use_bias=bias, key=key_fc1
        )
        self.w3 = eqx.nn.Linear(
            hidden_features, out_features, use_bias=bias, key=key_fc2
        )

        self.drop1 = eqx.nn.Dropout(dropout_rate)
        self.drop2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key_dr1, key_dr2 = jr.split(key, 2)

        x12 = jax.vmap(self.w12)(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        x = self.drop1(
            jax.nn.silu(x1) * x2,
            inference=inference,
            key=key_dr1,
        )

        x = self.drop2(
            jax.vmap(self.w3)(x),
            inference=inference,
            key=key_dr2,
        )

        return x


class SwiGluFused(SwiGlu):
    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        out_features: int | None = None,
        hidden_features: int | None = None,
        dropout_rate: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        """This matches the implementation of Dinov2 giant at
        https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/layers/swiglu_ffn.py#L54
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        super().__init__(
            in_features,
            key=key,
            out_features=out_features,
            hidden_features=hidden_features,
            dropout_rate=dropout_rate,
            bias=bias,
            **kwargs,
        )


def get_ffn(module: str | eqx.Module) -> eqx.Module:
    """Get an `eqx.Module` from its common name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    match module:
        case "mlp":
            return Mlp
        case "swiglu":
            return SwiGlu
        case "swiglufused":
            return SwiGluFused
        case "dinohead":
            return DINOHead
        case "weightnormlinear":
            return WeightNormLinear
        case _:
            raise ValueError(f"Got an unknown module string: {module}")
