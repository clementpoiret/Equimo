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

    weight_v: Float[Array, "out_features in_features"]
    weight_g: Float[Array, "out_features 1"]

    def __init__(self, in_features: int, out_features: int, key: PRNGKeyArray):
        self.weight_v = eqx.nn.Linear(
            in_features, out_features, use_bias=False, key=key
        ).weight
        self.weight_g = jnp.ones((out_features, 1))

    def __call__(
        self,
        x: Float[Array, "seqlen in_features"],
    ) -> Float[Array, "seqlen out_features"]:
        """Apply weight normalized linear transformation.

        Args:
            x: Input tensor of shape (seqlen, in_features)

        Returns:
            Transformed tensor of shape (seqlen, out_features)
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

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    last: WeightNormLinear
    act_layer: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features=2048,
        bottleneck_features=256,
        key: PRNGKeyArray,
        act_layer: Callable = jax.nn.gelu,
        **kwargs,
    ):
        """Initialize the DINOv2 projection head.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            hidden_features: Number of hidden features (default: 2048)
            bottleneck_features: Number of bottleneck features (default: 256)
            key: PRNG key for initialization
            act_layer: Activation function (default: gelu)
            **kwargs: Additional arguments
        """
        key_fc1, key_fc2, key_fc3, key_last = jr.split(key, 4)

        self.act_layer = act_layer

        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=key_fc1)
        self.fc2 = eqx.nn.Linear(hidden_features, hidden_features, key=key_fc2)
        self.fc3 = eqx.nn.Linear(hidden_features, bottleneck_features, key=key_fc3)
        self.last = WeightNormLinear(bottleneck_features, out_features, key=key_last)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        inference: Optional[bool] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seqlen out_features"]:
        """Process input through the DINOv2 projection head.

        Args:
            x: Input feature tensor of shape (seqlen, dim)
            inference: Unused; present for API consistency with other FFN modules
            key: Unused; present for API consistency with other FFN modules

        Returns:
            Projected and normalized features of shape (seqlen, out_features)
        """
        # Use larger eps for reduced-precision dtypes to avoid division by zero
        eps = 1e-6 if x.dtype in (jnp.float16, jnp.bfloat16) else 1e-12

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
    input -> fc1 -> activation -> norm -> dropout1 -> fc2 -> dropout2 -> output

    Note: the optional inter-layer norm is applied *after* the activation, following
    the NormFormer (post-activation norm) convention. See:
    https://arxiv.org/abs/2110.09456

    Attributes:
        fc1: First linear layer
        fc2: Second linear layer
        norm: Optional norm between fc1 and fc2 (applied post-activation)
        drop1: Dropout after first layer
        drop2: Dropout after second layer
        act_layer: Activation function
    """

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    norm: eqx.Module
    drop1: eqx.nn.Dropout
    drop2: eqx.nn.Dropout
    act_layer: Callable = eqx.field(static=True)

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
            norm_layer: Optional norm layer applied post-activation after fc1 (default: None)
            dropout_rate: Dropout probability (default: 0.0)
            bias: Whether to include bias in linear layers (default: True)
            eps: Epsilon for norm layer (default: 1e-5)
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
        mask: Optional[Float[Array, "seqlen 1"]] = None,
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

    Attributes:
        w1, w2: projection layers for both paths
        w3: Final projection layer
        drop1: Dropout after gating
        drop2: Dropout after final projection

    References:
        [1]: https://arxiv.org/pdf/2002.05202
    """

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
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
        align_to: int = 8,
        bias: bool = True,
        **kwargs,
    ):
        """Initialize the SwiGLU module.

        Args:
            in_features: Number of input features
            key: PRNG key for initialization
            out_features: Number of output features (default: same as in_features)
            hidden_features: Size of hidden dimension before alignment (default: same as in_features)
            dropout_rate: Dropout probability (default: 0.0)
            align_to: Constrains hidden features to be a multiple of this value (default: 8)
            bias: Whether to include bias in linear layers (default: True)
            **kwargs: Additional arguments (norm_layer is not supported and will be ignored)
        """
        key_fc1, key_fc2, key_fc3 = jr.split(key, 3)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        hidden_features = d + (-d % align_to)

        self.w1 = eqx.nn.Linear(
            in_features, hidden_features, use_bias=bias, key=key_fc1
        )
        self.w2 = eqx.nn.Linear(
            in_features, hidden_features, use_bias=bias, key=key_fc2
        )
        self.w3 = eqx.nn.Linear(
            hidden_features, out_features, use_bias=bias, key=key_fc3
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

        x1 = jax.vmap(self.w1)(x)
        x2 = jax.vmap(self.w2)(x)

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


class SwiGluFused(eqx.Module):
    """SwiGLU activation module with a fused projection, with dropout.

    This matches the implementation of Dinov2 giant at
    https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/layers/swiglu_ffn.py#L54

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
        align_to: int = 8,
        bias: bool = True,
        **kwargs,
    ):
        """Initialize the fused SwiGLU module.

        Args:
            in_features: Number of input features
            key: PRNG key for initialization
            out_features: Number of output features (default: same as in_features)
            hidden_features: Size of hidden dimension before alignment (default: same as in_features)
            dropout_rate: Dropout probability (default: 0.0)
            align_to: Constrains hidden features to be a multiple of this value (default: 8)
            bias: Whether to include bias in linear layers (default: True)
            **kwargs: Additional arguments (norm_layer is not supported and will be ignored)
        """
        key_fc1, key_fc2 = jr.split(key, 2)

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        hidden_features = d + (-d % align_to)

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


def get_ffn(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get an `eqx.Module` class from its common name.

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
