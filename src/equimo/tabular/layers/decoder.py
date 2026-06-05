from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.layers.activation import get_act

from .attention import SoftmaxScaling, _to_heads
from .registry import _register_module, _registry_name, _resolve_from_registry

_DECODER_REGISTRY: dict[str, type[eqx.Module]] = {}
_EMBEDDING_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_decoder(
    name: str | None = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a tabular decoder layer class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        registry_name = _registry_name(cls, name)
        return _register_module(
            _DECODER_REGISTRY,
            cls,
            registry_name,
            force,
            "tabular decoder",
            add_to_layer_registry=True,
        )

    return decorator


def get_decoder(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a tabular decoder layer class."""
    return _resolve_from_registry(module, _DECODER_REGISTRY, "tabular decoder")


def register_embedding(
    name: str | None = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a tabular embedding layer class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        registry_name = _registry_name(cls, name)
        return _register_module(
            _EMBEDDING_REGISTRY,
            cls,
            registry_name,
            force,
            "tabular embedding",
            add_to_layer_registry=True,
        )

    return decorator


def get_embedding(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a tabular embedding layer class."""
    return _resolve_from_registry(module, _EMBEDDING_REGISTRY, "tabular embedding")


@register_embedding()
class LabelEmbedding(eqx.Module):
    """Embedding layer for integer class labels."""

    embedding: eqx.nn.Embedding

    def __init__(self, num_classes: int, dim: int, *, key: PRNGKeyArray) -> None:
        self.embedding = eqx.nn.Embedding(num_classes, dim, key=key)

    def __call__(self, y: Int[Array, " rows"]) -> Float[Array, "rows dim"]:
        return jax.vmap(self.embedding)(y)


@register_embedding()
class LinearLabelEmbedding(eqx.Module):
    """Linear embedding layer for scalar regression targets."""

    projection: eqx.nn.Linear

    def __init__(self, num_outputs: int, dim: int, *, key: PRNGKeyArray) -> None:
        del num_outputs
        self.projection = eqx.nn.Linear(1, dim, key=key)

    def __call__(self, y: Float[Array, " rows"]) -> Float[Array, "rows dim"]:
        return jax.vmap(self.projection)(jnp.asarray(y).reshape(-1, 1))


@register_decoder()
class AttentionDecoder(eqx.Module):
    """Decode test-row logits by attending to labelled train-row embeddings."""

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    softmax_scaling: SoftmaxScaling | None
    num_classes: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        num_classes: int,
        dim: int,
        *,
        key: PRNGKeyArray,
        head_dim: int = 64,
        num_heads: int = 6,
        use_softmax_scaling: bool = True,
        scaling_mlp_hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        del kwargs
        key_q, key_k, key_scaling = jr.split(key, 3)
        inner_dim = head_dim * num_heads
        self.q_proj = eqx.nn.Linear(dim, inner_dim, key=key_q)
        self.k_proj = eqx.nn.Linear(dim, inner_dim, key=key_k)
        self.softmax_scaling = (
            SoftmaxScaling(
                num_heads,
                head_dim,
                scaling_mlp_hidden_dim,
                key=key_scaling,
            )
            if use_softmax_scaling
            else None
        )
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(
        self,
        train_embeddings: Float[Array, "train dim"],
        test_embeddings: Float[Array, "test dim"],
        targets: Int[Array, " train"],
        n_train: int,
    ) -> Float[Array, "test classes"]:
        q = _to_heads(
            self.q_proj,
            test_embeddings,
            self.num_heads,
            self.head_dim,
        )
        k = _to_heads(
            self.k_proj,
            train_embeddings,
            self.num_heads,
            self.head_dim,
        )
        if self.softmax_scaling is not None:
            q = self.softmax_scaling(q, n_train)

        one_hot = jax.nn.one_hot(targets, self.num_classes)
        scores = jnp.einsum("hmd,hnd->hmn", q, k) / jnp.sqrt(self.head_dim)
        attn = jax.nn.softmax(scores, axis=-1)
        probs = jnp.einsum("hmn,nc->hmc", attn, one_hot).mean(0)
        return jnp.log(jnp.clip(probs, min=1e-5) + 3e-5)


@register_decoder()
class RegressionDecoder(eqx.Module):
    """Decode test-row regression bucket logits from test embeddings."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    act_layer: Callable = eqx.field(static=True)

    def __init__(
        self,
        num_outputs: int,
        dim: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 2.0,
        act_layer: str | Callable = "exactgelu",
        **kwargs,
    ) -> None:
        del kwargs
        key_fc1, key_fc2 = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(dim, int(dim * mlp_ratio), key=key_fc1)
        self.fc2 = eqx.nn.Linear(int(dim * mlp_ratio), num_outputs, key=key_fc2)
        self.act_layer = get_act(act_layer)

    def __call__(
        self,
        train_embeddings: Float[Array, "train dim"],
        test_embeddings: Float[Array, "test dim"],
        targets: Array,
        n_train: int,
    ) -> Float[Array, "test outputs"]:
        del train_embeddings, targets, n_train
        return jax.vmap(lambda x: self.fc2(self.act_layer(self.fc1(x))))(
            test_embeddings
        )
