from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.dropout import Dropout


class Mlp(eqx.Module):
    """Multi-layer Perceptron (MLP) module with dropout.

    A standard MLP implementation with two fully connected layers, activation function,
    and dropout for regularization. The architecture follows:
    input -> fc1 -> activation -> dropout1 -> fc2 -> dropout2 -> output

    Attributes:
        fc1: First linear layer
        fc2: Second linear layer
        drop1: Dropout after first layer
        drop2: Dropout after second layer
        act_layer: Activation function
    """

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    drop1: Dropout
    drop2: Dropout
    act_layer: Callable

    def __init__(
        self,
        in_features: int,
        *,
        key: PRNGKeyArray,
        out_features: int | None = None,
        hidden_features: int | None = None,
        act_layer: Callable = jax.nn.gelu,
        dropout_rate: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        """Initialize the MLP.

        Args:
            in_features: Number of input features
            key: PRNG key for initialization
            out_features: Number of output features (default: same as in_features)
            hidden_features: Number of hidden features (default: same as in_features)
            act_layer: Activation function (default: gelu)
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
        self.fc2 = eqx.nn.Linear(
            hidden_features, out_features, use_bias=bias, key=key_fc2
        )

        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "seqlen dim"]:
        key_dr1, key_dr2 = jr.split(key, 2)

        x = self.drop1(
            self.act_layer(jax.vmap(self.fc1)(x)),
            inference=not enable_dropout,
            key=key_dr1,
        )
        x = self.drop2(
            jax.vmap(self.fc2)(x),
            inference=not enable_dropout,
            key=key_dr2,
        )

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
    drop1: Dropout
    drop2: Dropout

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
            in_features, hidden_features, use_bias=bias, key=key_fc1
        )
        self.w3 = eqx.nn.Linear(
            hidden_features // 2, out_features, use_bias=bias, key=key_fc2
        )

        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, "seqlen dim"]:
        key_dr1, key_dr2 = jr.split(key, 2)

        x12 = jax.vmap(self.w12)(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        x = self.drop1(
            jax.nn.silu(x1) * x2,
            inference=not enable_dropout,
            key=key_dr1,
        )

        x = self.drop2(
            jax.vmap(self.w3)(x),
            inference=not enable_dropout,
            key=key_dr2,
        )

        return x
