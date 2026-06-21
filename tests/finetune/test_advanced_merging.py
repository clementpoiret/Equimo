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


def test_ties_default_global_density_scope_differs_from_per_tensor():
    vector = eqft.TaskVector(
        delta={
            "large": jnp.array([100.0, 1.0]),
            "small": jnp.array([0.9, 0.8]),
        },
        include_head=True,
        base_architecture_hash="arch",
        base_checkpoint_hash="base",
    )

    global_merged = eqft.ties_merge([vector], config=eqft.TIESConfig(density=0.5))
    per_tensor_merged = eqft.ties_merge(
        [vector],
        config=eqft.TIESConfig(density=0.5, density_scope="per_tensor"),
    )

    assert jnp.allclose(global_merged.delta["large"], jnp.array([100.0, 1.0]))
    assert jnp.allclose(global_merged.delta["small"], jnp.array([0.0, 0.0]))
    assert jnp.allclose(per_tensor_merged.delta["large"], jnp.array([100.0, 0.0]))
    assert jnp.allclose(per_tensor_merged.delta["small"], jnp.array([0.9, 0.0]))


def test_dare_config_seed_matches_explicit_key():
    vector = eqft.TaskVector(
        delta=(jnp.arange(6.0),),
        include_head=True,
        base_architecture_hash="arch",
        base_checkpoint_hash="base",
    )

    from_config = eqft.dare_task_vector(
        vector,
        config=eqft.DARETransform(drop_rate=0.5, seed=7),
    )
    from_key = eqft.dare_task_vector(vector, drop_rate=0.5, key=jr.PRNGKey(7))

    assert jnp.array_equal(from_config.delta[0], from_key.delta[0])


def test_dare_global_scope_uses_one_flat_mask():
    vector = eqft.TaskVector(
        delta=(jnp.arange(1.0, 5.0), jnp.arange(5.0, 9.0)),
        include_head=True,
        base_architecture_hash="arch",
        base_checkpoint_hash="base",
    )

    masked = eqft.dare_task_vector(
        vector,
        config=eqft.DARETransform(
            drop_rate=0.5,
            rescale=False,
            seed=11,
            scope="global",
        ),
    )
    mask = jr.bernoulli(jr.PRNGKey(11), 0.5, (8,))

    assert jnp.array_equal(masked.delta[0], jnp.where(mask[:4], vector.delta[0], 0))
    assert jnp.array_equal(masked.delta[1], jnp.where(mask[4:], vector.delta[1], 0))


def test_dare_requires_key_or_config_seed():
    vector = eqft.TaskVector(delta=(jnp.arange(3.0),))

    with pytest.raises(ValueError, match="key or DARETransform.seed"):
        eqft.dare_task_vector(vector)


def test_breadcrumbs_masks_middle_magnitudes_and_validates_config():
    vector = eqft.TaskVector(delta=jnp.array([0.1, 0.2, 1.0, 100.0]))

    masked = eqft.breadcrumbs_task_vector(
        vector,
        bottom_fraction=0.25,
        top_fraction=0.25,
    )

    assert jnp.allclose(masked.delta, jnp.array([0.0, 0.2, 1.0, 0.0]))
    with pytest.raises(ValueError, match="rescale"):
        eqft.breadcrumbs_task_vector(
            vector,
            config=eqft.BreadcrumbsConfig(rescale=True),
        )
    with pytest.raises(ValueError, match="less than 1"):
        eqft.breadcrumbs_task_vector(
            vector,
            config=eqft.BreadcrumbsConfig(bottom_fraction=0.5, top_fraction=0.5),
        )


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


def test_fisher_merge_normalizes_fisher_trees_when_configured():
    models = [
        {"w": jnp.array([0.0]), "b": jnp.array([0.0])},
        {"w": jnp.array([10.0]), "b": jnp.array([10.0])},
    ]
    fishers = [
        {"w": jnp.array([100.0]), "b": jnp.array([0.0])},
        {"w": jnp.array([1.0]), "b": jnp.array([0.0])},
    ]

    normalized = eqft.fisher_merge(models, fishers)
    raw = eqft.fisher_merge(
        models,
        fishers,
        config=eqft.FisherMergeConfig(normalize_fisher=False),
    )

    assert jnp.allclose(normalized["w"], jnp.array([5.0]))
    assert jnp.allclose(normalized["b"], jnp.array([0.0]))
    assert jnp.allclose(raw["w"], jnp.array([10.0 / 101.0]))


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


def test_regmean_alternate_solvers_match_solve():
    model_a = jnp.array([[1.0, 2.0]])
    model_b = jnp.array([[3.0, 4.0]])
    covariance_a = jnp.array([[2.0, 0.2], [0.2, 1.5]])
    covariance_b = jnp.array([[1.0, 0.1], [0.1, 1.0]])

    expected = eqft.regmean_merge(
        [model_a, model_b],
        [covariance_a, covariance_b],
        config=eqft.RegMeanConfig(solver="solve"),
    )

    for solver in ("cholesky", "svd"):
        merged = eqft.regmean_merge(
            [model_a, model_b],
            [covariance_a, covariance_b],
            config=eqft.RegMeanConfig(solver=solver),
        )
        assert jnp.allclose(merged, expected, atol=1e-5)


def test_regmean_non_matrix_leaves_use_mean_policy():
    merged = eqft.regmean_merge(
        [
            {"bias": jnp.array([1.0, 3.0])},
            {"bias": jnp.array([3.0, 5.0])},
        ],
        [
            {"bias": jnp.array([1.0, 100.0])},
            {"bias": jnp.array([100.0, 1.0])},
        ],
    )

    assert jnp.allclose(merged["bias"], jnp.array([2.0, 4.0]))


def test_regmean_rejects_unsupported_config_variants():
    model = jnp.array([[1.0, 2.0]])
    covariance = jnp.eye(2)

    with pytest.raises(ValueError, match="covariance_normalization"):
        eqft.regmean_merge(
            [model, model],
            [covariance, covariance],
            config=eqft.RegMeanConfig(covariance_normalization="sample_mean"),
        )

    with pytest.raises(ValueError, match="covariance leaves"):
        eqft.regmean_merge(
            [model, model],
            [model, model],
            config=eqft.RegMeanConfig(non_matrix_policy="error"),
        )
