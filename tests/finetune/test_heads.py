"""Fine-tuning head tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


class KeyInferenceHead(eqx.Module):
    def __call__(self, x, *, key, inference: bool | None = True):
        inference_flag = 0.0 if inference else 1.0
        return jnp.asarray([jnp.sum(x), inference_flag, jr.uniform(key, ())])


def test_multilabel_head_raw_logits():
    key = jr.PRNGKey(0)
    head = eqft.MultiLabelHead(4, 3, key=key)
    x = jnp.ones((4,))

    logits = head(x)

    assert logits.shape == (3,)
    assert jnp.any((logits < 0.0) | (logits > 1.0))


def test_ctc_head_frame_logits():
    key = jr.PRNGKey(1)
    head = eqft.CTCHead(4, 7, key=key)
    x = jnp.ones((5, 4))

    logits = head(x)

    assert logits.shape == (5, 7)
    assert head.blank_id == 0


def test_contrastive_projection_head_normalizes():
    key = jr.PRNGKey(2)
    head = eqft.ContrastiveProjectionHead(4, 3, key=key)
    x = jnp.ones((4,))

    y = head(x)

    assert y.shape == (3,)
    assert jnp.linalg.norm(y) == jnp.array(1.0)


def test_dense_feature_adapter_projects_last_axis():
    key = jr.PRNGKey(3)
    adapter = eqft.DenseFeatureAdapter(4, 2, key=key, activation="relu")
    x = jnp.ones((3, 5, 4))

    y = adapter(x)

    assert y.shape == (3, 5, 2)


def test_layer_norm_readout_head_shapes():
    key = jr.PRNGKey(4)
    head = eqft.LayerNormReadoutHead(4, eqft.LinearHead(4, 3, key=key))

    y = head(jnp.arange(4, dtype=jnp.float32))
    y_batched = head(jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4))

    assert y.shape == (3,)
    assert y_batched.shape == (2, 3, 3)


def test_layer_norm_readout_head_forwards_key_and_inference():
    head = eqft.LayerNormReadoutHead(4, KeyInferenceHead())

    y = head(
        jnp.arange(4, dtype=jnp.float32),
        key=jr.PRNGKey(5),
        inference=False,
    )

    assert y.shape == (3,)
    assert y[1] == jnp.array(1.0)
