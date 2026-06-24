"""Encode token IDs with the lightweight text transformer API."""

import jax.numpy as jnp
import jax.random as jr

from equimo.language import TextTransformerEncoder


def main() -> None:
    key = jr.PRNGKey(0)
    model_key, forward_key = jr.split(key)

    model = TextTransformerEncoder(
        dim=16,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=128,
        key=model_key,
    )

    token_ids = jnp.array([12, 7, 91, 4, 0, 0])
    padding = jnp.array([0, 0, 0, 0, 1, 1])

    token_features = model.features(token_ids, padding, key=forward_key, inference=True)
    pooled_embedding = model(token_ids, padding, key=forward_key, inference=True)

    print("token_features:", token_features.shape)
    print("pooled_embedding:", pooled_embedding.shape)


if __name__ == "__main__":
    main()
