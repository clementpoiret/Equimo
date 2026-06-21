"""Advanced merging tests."""

from __future__ import annotations

import pytest
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def test_ties_dare_breadcrumbs_task_vectors(tiny_vision_transformer):
    tuned_a = eqft.interpolate_models(
        tiny_vision_transformer,
        tiny_vision_transformer,
        alpha=0.0,
        include_head=True,
    )
    tuned_b = eqft.interpolate_models(
        tiny_vision_transformer,
        tiny_vision_transformer,
        alpha=0.0,
        include_head=True,
    )
    vector_a = eqft.task_vector(tiny_vision_transformer, tuned_a, include_head=True)
    vector_b = eqft.task_vector(tiny_vision_transformer, tuned_b, include_head=True)

    ties = eqft.ties_merge([vector_a, vector_b], density=1.0)
    dare = eqft.dare_task_vector(ties, drop_rate=0.0, key=jr.PRNGKey(0))
    breadcrumbs = eqft.breadcrumbs_task_vector(dare)

    assert breadcrumbs.include_head is True
    assert jnp.array_equal(breadcrumbs.delta.head.weight, vector_a.delta.head.weight)


def test_fisher_and_regmean_require_external_statistics(tiny_vision_transformer):
    with pytest.raises(ValueError, match="Fisher statistics"):
        eqft.fisher_merge([tiny_vision_transformer])

    with pytest.raises(ValueError, match="covariance statistics"):
        eqft.regmean_merge([tiny_vision_transformer])


def test_fisher_merge_uses_supplied_diagonal_statistics():
    merged = eqft.fisher_merge(
        [jnp.array([1.0, 3.0]), jnp.array([3.0, 5.0])],
        [jnp.array([1.0, 3.0]), jnp.array([3.0, 1.0])],
    )

    assert jnp.allclose(merged, jnp.array([2.5, 3.5]))


def test_regmean_merge_uses_input_covariances():
    model_a = jnp.array([[1.0, 2.0]])
    model_b = jnp.array([[3.0, 4.0]])
    covariance_a = jnp.eye(2) * 2.0
    covariance_b = jnp.eye(2)

    merged = eqft.regmean_merge(
        [model_a, model_b],
        [covariance_a, covariance_b],
        ridge=0.0,
    )

    assert jnp.allclose(merged, jnp.array([[5.0 / 3.0, 8.0 / 3.0]]))
