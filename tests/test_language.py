import jax.numpy as jnp
import jax.random as jr

from equimo.language.models import TextTransformerEncoder
from equimo.registry import get_model_cls


KEY = jr.PRNGKey(0)


def test_text_transformer_encoder_forward_and_features_shape():
    model = TextTransformerEncoder(
        dim=16,
        mlp_ratio=2.0,
        depth=2,
        num_heads=2,
        vocab_size=128,
        key=KEY,
    )
    ids = jnp.array([1, 2, 3, 0, 0])
    padding_mask = jnp.array([0, 0, 0, 1, 1])

    features = model.features(ids, padding_mask, key=KEY, inference=True)
    pooled = model(ids, padding_mask, key=KEY, inference=True)

    assert features.shape == (5, 16)
    assert pooled.shape == (16,)
    assert jnp.all(jnp.isfinite(features))
    assert jnp.all(jnp.isfinite(pooled))


def test_language_model_registered_by_modality():
    assert (
        get_model_cls("text_transformer_encoder", modality="language")
        is TextTransformerEncoder
    )
