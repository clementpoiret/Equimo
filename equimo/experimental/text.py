from pathlib import Path
from typing import Callable, Optional, Sequence
from urllib.parse import urlparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import requests
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.attention import AttentionBlock

try:
    import tensorflow as tf
    import tensorflow_text
except ImportError:
    print(
        "`tensorflow` and `tensorflow_text` need to be installed to use tokenizers. Install equimo with the `text` group such as `uv add equimo[text]`"
    )

DEFAULT_TOKENIZER_REPOSITORY = (
    "https://huggingface.co/poiretclement/equimo/resolve/main/models/tokenizers"
)


class Tokenizer(object):
    """A simple tokenizer based on TensorFlow's SentencepieceTokenizer.

    This class provides a wrapper around TensorFlow's SentencepieceTokenizer,
    handling model loading from a path (which can be a URL or local file) and
    tokenizing input text with padding. It supports downloading remote models and
    caching them locally.

    Attributes:
        tokenizer: The underlying TensorFlow SentencepieceTokenizer instance.

    Based on: https://github.com/google-deepmind/tips/blob/72820c1841f973c9543d9c95c5ff2262ec621955/pytorch/text_encoder.py#L27
    """

    def __init__(
        self,
        identifier: Optional[str] = None,
        url: Optional[str] = None,
        path: Optional[str] = None,
        repository: str = DEFAULT_TOKENIZER_REPOSITORY,
    ):
        """Initializes the tokenizer."""
        if not identifier and not path and not url:
            raise ValueError(
                "At least one of identifier, path, or url should be defined."
            )

        if identifier:
            url = f"{repository}/{identifier}.model"
        if url:
            model_file_path = self.download(url)
        else:
            # User passes a local file
            assert path
            path: Path = Path(path)
            if not path.expanduser().exists():
                raise FileNotFoundError(f"Tokenizer file not found: {path}")
            model_file_path = path

        with open(model_file_path, "rb") as f:
            model = f.read()
        self.tokenizer = tensorflow_text.SentencepieceTokenizer(
            model=model, add_eos=False, add_bos=False
        )

    def download(self, url: str) -> Path:
        assert url.startswith("http://") or url.startswith("https://")

        parsed_url = urlparse(url)
        fname = Path(parsed_url.path).name
        cache_dir = Path("~/.cache/equimo/tokenizers").expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file_path = cache_dir / fname

        if not model_file_path.exists():
            response = requests.get(url)
            response.raise_for_status()
            with open(model_file_path, "wb") as f:
                f.write(response.content)

        return model_file_path

    def tokenize(self, input_text, max_len=64):
        tokens = self.tokenizer.tokenize(tf.strings.lower(input_text)).to_tensor()
        curr_len = tokens.shape[1]
        is_padding = tf.zeros((tokens.shape[0], max_len))
        if curr_len > max_len:
            tokens = tokens[:, :max_len]
        else:
            padding_len = max_len - curr_len
            tokens = tf.pad(tokens, [[0, 0], [0, padding_len]], constant_values=0)
            is_padding = tf.cast(tokens == 0, tf.int32)
        return tokens.numpy(), is_padding.numpy()


def global_avg_pooling(
    inputs: Float[Array, "..."],
    compatible_paddings: Int[Array, "..."],
    pooling_dims: Sequence[int],
    epsilon: float = 1e-8,
):
    """
    Applies global average pooling to inputs over specified dimensions, handling padding.

    Args:
        inputs: Input array of shape [...].
        compatible_paddings: Padding mask, same shape as inputs, where >0 indicates padded.
        pooling_dims: Sequence of int, dimensions to pool over.
        epsilon: Small float to prevent division by zero, default 1e-8.

    Returns:
        Array with pooling_dims reduced, averaged over valid elements.
    """
    valid_mask = 1.0 - compatible_paddings
    masked_inputs = inputs * valid_mask
    inputs_sum = jnp.sum(masked_inputs, axis=pooling_dims)
    valid_count = jnp.sum(valid_mask, axis=pooling_dims)
    outputs = inputs_sum / (valid_count + epsilon)
    return outputs


class Transformer(eqx.Module):
    """A transformer model composed of a stack of attention blocks.

    This class implements a transformer encoder by stacking multiple `AttentionBlock`
    instances, suitable for processing sequential data such as text embeddings.

    Attributes:
        blocks: A list of `AttentionBlock` instances forming the transformer stack.
    """

    blocks: list[AttentionBlock]

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
        """Initializes the Transformer with a stack of attention blocks.

        Args:
            dim: The dimension of the input and output embeddings.
            depth: The number of attention blocks in the transformer stack.
            num_heads: The number of attention heads in each block.
            mlp_ratio: The ratio of the MLP hidden dimension to the embedding dimension.
            key: A PRNG key for initializing the attention blocks.
            act_layer: The activation function for the MLP layers. Can be a callable or a
            string name of an activation function (default: jax.nn.gelu).
        """

        keys = jr.split(key, depth)

        act_layer = get_act(act_layer)

        self.blocks = [
            AttentionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                key=keys[i],
            )
            for i in range(depth)
        ]

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        mask: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "seqlen dim"]:
        for block in self.blocks:
            x = block(x, mask=mask, inference=inference, key=key)
        return x


class TextEncoder(eqx.Module):
    """A text encoder model that processes token sequences into fixed-size embeddings.

    This class combines token embedding, positional embedding, a transformer encoder,
    and final layer normalization, followed by global average pooling to produce a
    single embedding vector per sequence.

    Attributes:
        token_embedding: An `eqx.nn.Embedding` layer for converting token IDs to embeddings.
        transformer: A `Transformer` instance for processing the embedded sequences.
        ln_final: An `eqx.nn.LayerNorm` layer for normalizing the transformer output.
        dim: The dimension of the embeddings (static field).
        scale_sqrt_depth: Boolean indicating whether to scale embeddings by sqrt(dim) (static field).
        temperature: A temperature parameter, currently unused (static field, default: -1).
            While unused, this parameter may be useful at inference time.
    """

    token_embedding: eqx.nn.Embedding
    transformer: Transformer
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

        self.transformer = Transformer(
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
        """
        Generates a tensor of sinusoids with different frequencies for positional embeddings.

        Args:
            embedding_size: Dimension of the embedding to be generated.
            min_timescale: Start of the geometric index, determining the periodicity of the signal.
            max_timescale: End of the geometric index, determining the frequency of the signal.
            seq_len: Optional sequence length if position is not provided.
            position: Optional 1D array of positions for each token in the sequence.

        Returns:
            Positional embeddings of shape [seq_len, embedding_size] with dtype float32.

        Raises:
            ValueError: If neither position nor seq_len is provided, or if position is not a 1D array.
        """
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

        # Pad with a zero column if embedding_size is odd
        if self.dim % 2 == 1:
            signal = jnp.pad(
                signal, ((0, 0), (0, 1)), mode="constant", constant_values=0
            )

        return signal

    def __call__(
        self,
        ids: Int[Array, "seqlen"],
        paddings: Float[Array, "seqlen"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "dim"]:
        """Applies TextEncoder module."""
        seq_len = ids.shape[0]
        mask = (paddings == 0).astype(jnp.float32)

        x = jax.vmap(self.token_embedding)(ids)
        if self.scale_sqrt_depth:
            x = x * (self.dim**0.5)

        x = x + self.posemb(seq_len=seq_len)
        x = self.transformer(x, mask=mask[:, None], inference=inference, key=key)
        x = jax.vmap(self.ln_final)(x)
        x = global_avg_pooling(
            x, compatible_paddings=paddings[:, None], pooling_dims=[0]
        )

        return x
