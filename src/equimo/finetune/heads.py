"""Task heads for Equimo fine-tuning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


ActivationName = Literal["gelu", "relu", "silu", "tanh", "identity"]


class IdentityHead(eqx.Module):
    """Head that returns inputs unchanged."""

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        del kwargs
        return x


class LinearHead(eqx.Module):
    """Linear task head with Equimo's fine-tuning defaults."""

    linear: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        bias: bool = True,
        weight_init: str = "trunc_normal_0.02",
        bias_init: float = 0.0,
    ):
        self.linear = eqx.nn.Linear(
            in_features,
            out_features,
            use_bias=bias,
            key=key,
        )
        self.linear = _init_linear(
            self.linear,
            key,
            weight_init=weight_init,
            bias_init=bias_init,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return _apply_last_axis(self.linear, x)


class MultiLabelHead(eqx.Module):
    """Linear multi-label head that returns raw logits."""

    head: LinearHead

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        bias: bool = True,
        bias_prior: float | None = None,
    ):
        bias_init = 0.0 if bias_prior is None else _logit(bias_prior)
        self.head = LinearHead(
            in_features,
            out_features,
            key=key,
            bias=bias,
            bias_init=bias_init,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(x)


class MLPHead(eqx.Module):
    """Small MLP task head."""

    layers: tuple[eqx.nn.Linear, ...]
    activation: ActivationName = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        if num_layers < 1:
            raise ValueError("MLPHead requires num_layers >= 1.")

        hidden_dim = in_features if hidden_dim is None else hidden_dim
        keys = jr.split(key, num_layers)
        dims = [in_features]
        if num_layers > 1:
            dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(out_features)

        self.layers = tuple(
            _init_linear(
                eqx.nn.Linear(dims[i], dims[i + 1], use_bias=bias, key=keys[i]),
                keys[i],
            )
            for i in range(num_layers)
        )
        self.activation = activation
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        dropout_keys = (
            jr.split(key, max(len(self.layers) - 1, 1)) if key is not None else ()
        )
        for index, layer in enumerate(self.layers):
            x = _apply_last_axis(layer, x)
            if index == len(self.layers) - 1:
                continue
            x = _activation(self.activation)(x)
            if self.dropout > 0.0 and not inference:
                if key is None:
                    raise ValueError(
                        "A PRNG key is required when MLPHead dropout is active."
                    )
                x = _dropout(x, self.dropout, dropout_keys[index])
        return x


class ProjectionHead(eqx.Module):
    """Projection MLP used by contrastive or embedding tasks."""

    head: MLPHead

    def __init__(
        self,
        in_features: int,
        out_dim: int,
        *,
        key: jax.Array,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
    ):
        self.head = MLPHead(
            in_features,
            out_dim,
            key=key,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        return self.head(x, key=key, inference=inference)


class ContrastiveProjectionHead(eqx.Module):
    """Projection head with optional L2 normalization."""

    projection: ProjectionHead
    l2_normalize: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_dim: int,
        *,
        key: jax.Array,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        l2_normalize: bool = True,
        eps: float = 1e-12,
    ):
        self.projection = ProjectionHead(
            in_features,
            out_dim,
            key=key,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )
        self.l2_normalize = l2_normalize
        self.eps = eps

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        x = self.projection(x, key=key, inference=inference)
        if not self.l2_normalize:
            return x
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / jnp.maximum(norm, self.eps)


class CTCHead(eqx.Module):
    """Frame-level CTC head that returns raw logits."""

    head: LinearHead
    blank_id: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        vocab_size: int,
        *,
        key: jax.Array,
        blank_id: int = 0,
        bias: bool = True,
    ):
        self.head = LinearHead(in_features, vocab_size, key=key, bias=bias)
        self.blank_id = blank_id

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(x)


class TokenClassificationHead(eqx.Module):
    """Token-level classification head that returns raw logits."""

    head: LinearHead

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        bias: bool = True,
    ):
        self.head = LinearHead(in_features, out_features, key=key, bias=bias)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(x)


class DenseFeatureAdapter(eqx.Module):
    """Project dense or token features along the last axis."""

    projection: eqx.nn.Linear
    activation: ActivationName = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        activation: ActivationName = "identity",
        dropout: float = 0.0,
        bias: bool = True,
        weight_init: str = "trunc_normal_0.02",
        bias_init: float = 0.0,
    ):
        self.projection = _init_linear(
            eqx.nn.Linear(
                in_features,
                out_features,
                use_bias=bias,
                key=key,
            ),
            key,
            weight_init=weight_init,
            bias_init=bias_init,
        )
        self.activation = activation
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        y = _apply_last_axis(self.projection, x)
        y = _activation(self.activation)(y)
        if self.dropout > 0.0 and not inference:
            if key is None:
                raise ValueError(
                    "A PRNG key is required when DenseFeatureAdapter dropout is active."
                )
            y = _dropout(y, self.dropout, key)
        return y


def _init_linear(
    linear: eqx.nn.Linear,
    key: jax.Array,
    *,
    weight_init: str = "trunc_normal_0.02",
    bias_init: float = 0.0,
) -> eqx.nn.Linear:
    if weight_init == "trunc_normal_0.02":
        weight = jr.truncated_normal(
            key,
            lower=-2.0,
            upper=2.0,
            shape=linear.weight.shape,
            dtype=linear.weight.dtype,
        ) * jnp.asarray(0.02, dtype=linear.weight.dtype)
    elif weight_init == "zeros":
        weight = jnp.zeros_like(linear.weight)
    else:
        raise ValueError(
            f"Unsupported weight_init {weight_init!r}; expected trunc_normal_0.02 or zeros."
        )

    linear = eqx.tree_at(lambda m: m.weight, linear, weight)
    if linear.bias is not None:
        bias = jnp.full_like(linear.bias, bias_init)
        linear = eqx.tree_at(lambda m: m.bias, linear, bias)
    return linear


def _apply_last_axis(
    module: Callable[[jax.Array], jax.Array], x: jax.Array
) -> jax.Array:
    if x.ndim == 1:
        return module(x)
    leading_shape = x.shape[:-1]
    x_flat = x.reshape((-1, x.shape[-1]))
    y_flat = jax.vmap(module)(x_flat)
    return y_flat.reshape((*leading_shape, y_flat.shape[-1]))


def _activation(name: ActivationName) -> Callable[[jax.Array], jax.Array]:
    if name == "gelu":
        return jax.nn.gelu
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    if name == "tanh":
        return jnp.tanh
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported activation {name!r}.")


def _dropout(x: jax.Array, rate: float, key: jax.Array) -> jax.Array:
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("bias_prior must be in the open interval (0, 1).")
    return float(jnp.log(p / (1.0 - p)))


__all__ = (
    "CTCHead",
    "ContrastiveProjectionHead",
    "DenseFeatureAdapter",
    "IdentityHead",
    "LinearHead",
    "MLPHead",
    "MultiLabelHead",
    "ProjectionHead",
    "TokenClassificationHead",
)
