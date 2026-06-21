"""Fine-tuning head tests."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


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
