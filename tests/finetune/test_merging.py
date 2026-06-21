"""Model merging tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

import equimo.finetune as eqft

from fixtures import assert_tree_allclose


def _shift_head(model, value: float):
    return eqx.tree_at(
        lambda m: m.head.weight,
        model,
        model.head.weight + value,
    )


def test_wise_ft_endpoints(tiny_vision_transformer):
    tuned = _shift_head(tiny_vision_transformer, 1.0)

    alpha0 = eqft.interpolate_models(
        tiny_vision_transformer,
        tuned,
        alpha=0.0,
        include_head=True,
    )
    alpha1 = eqft.interpolate_models(
        tiny_vision_transformer,
        tuned,
        alpha=1.0,
        include_head=True,
    )

    assert_tree_allclose(alpha0, tiny_vision_transformer)
    assert_tree_allclose(alpha1, tuned)


def test_uniform_soup_arithmetic_mean(tiny_vision_transformer):
    model_a = _shift_head(tiny_vision_transformer, 1.0)
    model_b = _shift_head(tiny_vision_transformer, 3.0)

    soup = eqft.uniform_soup([model_a, model_b], include_head=True)

    expected = tiny_vision_transformer.head.weight + 2.0
    assert jnp.allclose(soup.head.weight, expected)


def test_greedy_soup_uses_score_function(tiny_vision_transformer):
    calls = []

    def score_fn(model):
        calls.append(model)
        return float(jnp.sum(model.head.weight))

    models = [
        tiny_vision_transformer,
        _shift_head(tiny_vision_transformer, 1.0),
        _shift_head(tiny_vision_transformer, -10.0),
    ]

    _, selected = eqft.greedy_soup(models, score_fn)

    assert selected == (0, 1)
    assert len(calls) == 3


def test_task_vector_reconstructs_tuned_model(tiny_vision_transformer):
    tuned = _shift_head(tiny_vision_transformer, 1.0)
    vector = eqft.task_vector(tiny_vision_transformer, tuned, include_head=True)

    reconstructed = eqft.apply_task_vector(tiny_vision_transformer, vector, scale=1.0)

    assert_tree_allclose(reconstructed, tuned)
