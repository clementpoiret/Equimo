# ty: ignore[invalid-assignment]
from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.layers.activation import get_act
from equimo.core.layers.attention import AttentionBlock
from equimo.registry import register_model

__all__ = ["TextTransformerEncoder", "TransformerEncoderStack", "global_avg_pooling"]


def global_avg_pooling(
    inputs: Float[Array, "..."],
    compatible_paddings: Int[Array, "..."],
    pooling_dims: Sequence[int],
    epsilon: float = 1e-8,
):
    """Average unpadded tokens over the requested pooling dimensions."""

    valid_mask = 1.0 - compatible_paddings
    masked_inputs = inputs * valid_mask
    inputs_sum = jnp.sum(masked_inputs, axis=pooling_dims)
    valid_count = jnp.sum(valid_mask, axis=pooling_dims)
    return inputs_sum / (valid_count + epsilon)


class TransformerEncoderStack(eqx.Module):
    """Stack of modality-neutral transformer blocks for token sequences."""

    blocks: Tuple[AttentionBlock, ...]

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        key: PRNGKeyArray,
        act_layer: Callable | str = jax.nn.gelu,
    ):
        keys = jr.split(key, depth)
        act_layer = get_act(act_layer)
        self.blocks = tuple(
            AttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                key=keys[i],
            )
            for i in range(depth)
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        mask: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "seqlen dim"]:
        keys = jr.split(key, len(self.blocks))
        for block, block_key in zip(self.blocks, keys):
            x = block(x, mask=mask, inference=inference, key=block_key)
        return x


@register_model("text_transformer_encoder", modality="language")
class TextTransformerEncoder(eqx.Module):
    """Transformer text encoder that returns pooled sequence embeddings."""

    token_embedding: eqx.nn.Embedding
    transformer: TransformerEncoderStack
    ln_final: eqx.nn.LayerNorm

    dim: int = eqx.field(static=True)
    scale_sqrt_depth: bool = eqx.field(static=True)
    temperature: float = eqx.field(static=True, default=-1)

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        depth: int,
        num_heads: int,
        vocab_size: int,
        *,
        key: PRNGKeyArray,
        scale_sqrt_depth: bool = True,
        act_layer: Callable | str = jax.nn.gelu,
        temperature: float = -1.0,
    ):
        key_emb, key_trans = jr.split(key, 2)
        self.dim = dim
        self.scale_sqrt_depth = scale_sqrt_depth
        self.temperature = temperature
        self.token_embedding = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=dim,
            key=key_emb,
        )
        self.transformer = TransformerEncoderStack(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            key=key_trans,
        )
        self.ln_final = eqx.nn.LayerNorm(dim)

    def posemb(
        self,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        seq_len: Optional[int] = None,
        position: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if position is None:
            if seq_len is None:
                raise ValueError("If position is None, seq_len must be provided.")
            position = jnp.arange(seq_len, dtype=jnp.float32)
        elif position.ndim != 1:
            raise ValueError("position must be a 1D array.")

        num_timescales = self.dim // 2
        log_timescale_increment = jnp.log(max_timescale / min_timescale) / jnp.maximum(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * jnp.exp(
            jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
        )
        scaled_time = position[:, None] * inv_timescales[None, :]
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
        if self.dim % 2 == 1:
            signal = jnp.pad(signal, ((0, 0), (0, 1)), mode="constant")
        return signal

    def features(
        self,
        ids: Int[Array, "seqlen"],  # noqa: F821
        padding_mask: Float[Array, "seqlen"],  # noqa: F821
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        seq_len = ids.shape[0]
        valid_mask = (padding_mask == 0).astype(jnp.float32)

        x = jax.vmap(self.token_embedding)(ids)
        if self.scale_sqrt_depth:
            x = x * (self.dim**0.5)

        x = x + self.posemb(seq_len=seq_len)
        x = self.transformer(x, mask=valid_mask[:, None], inference=inference, key=key)
        return jax.vmap(self.ln_final)(x)

    def __call__(
        self,
        ids: Int[Array, "seqlen"],  # noqa: F821
        padding_mask: Float[Array, "seqlen"],  # noqa: F821
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "dim"]:
        x = self.features(ids, padding_mask, key=key, inference=inference)
        return global_avg_pooling(
            x, compatible_paddings=padding_mask[:, None], pooling_dims=[0]
        )
