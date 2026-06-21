"""Model merging tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

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


def test_greedy_soup_config_starts_from_best_model(tiny_vision_transformer):
    models = [
        tiny_vision_transformer,
        _shift_head(tiny_vision_transformer, 10.0),
        _shift_head(tiny_vision_transformer, 5.0),
    ]

    _, selected = eqft.greedy_soup(
        models,
        lambda model: float(jnp.sum(model.head.weight)),
        config=eqft.GreedySoupConfig(),
    )

    assert selected == (1,)


def test_task_vector_reconstructs_tuned_model(tiny_vision_transformer):
    tuned = _shift_head(tiny_vision_transformer, 1.0)
    vector = eqft.task_vector(tiny_vision_transformer, tuned, include_head=True)

    reconstructed = eqft.apply_task_vector(tiny_vision_transformer, vector, scale=1.0)

    assert_tree_allclose(reconstructed, tuned)


def test_wise_ft_config_controls_alpha_and_head_inclusion(tiny_vision_transformer):
    tuned = _shift_head(tiny_vision_transformer, 4.0)

    merged = eqft.interpolate_models(
        tiny_vision_transformer,
        tuned,
        config=eqft.WiSEFTConfig(alpha=0.25, include_head=True),
    )

    assert jnp.allclose(
        merged.head.weight,
        tiny_vision_transformer.head.weight + 1.0,
    )


def test_wise_ft_head_policy_controls_classifier_leaves(tiny_vision_transformer):
    tuned = _shift_head(tiny_vision_transformer, 4.0)

    zero_shot_head = eqft.interpolate_models(
        tiny_vision_transformer,
        tuned,
        config=eqft.WiSEFTConfig(
            alpha=1.0,
            include_head=True,
            head_policy="use_zero_shot",
        ),
    )
    finetuned_head = eqft.interpolate_models(
        tiny_vision_transformer,
        tuned,
        config=eqft.WiSEFTConfig(alpha=0.0, head_policy="use_finetuned"),
    )

    assert jnp.allclose(zero_shot_head.head.weight, tiny_vision_transformer.head.weight)
    assert jnp.allclose(finetuned_head.head.weight, tuned.head.weight)

    with pytest.raises(ValueError, match="head_policy"):
        eqft.interpolate_models(
            tiny_vision_transformer,
            tuned,
            config=eqft.WiSEFTConfig(head_policy="unknown"),
        )


def test_wise_ft_non_strict_shapes_skip_incompatible_leaves(tiny_vision_transformer):
    tuned = eqx.tree_at(
        lambda model: model.head.weight,
        tiny_vision_transformer,
        jnp.ones((3, 4), dtype=tiny_vision_transformer.head.weight.dtype),
    )

    with pytest.raises(ValueError, match="different shapes"):
        eqft.interpolate_models(
            tiny_vision_transformer,
            tuned,
            include_head=True,
        )

    merged = eqft.interpolate_models(
        tiny_vision_transformer,
        tuned,
        config=eqft.WiSEFTConfig(
            alpha=1.0,
            include_head=True,
            strict_shapes=False,
            require_same_architecture_hash=False,
        ),
    )

    assert jnp.array_equal(merged.head.weight, tiny_vision_transformer.head.weight)


def test_uniform_soup_config_controls_weights(tiny_vision_transformer):
    model_a = _shift_head(tiny_vision_transformer, 1.0)
    model_b = _shift_head(tiny_vision_transformer, 3.0)

    soup = eqft.uniform_soup(
        [model_a, model_b],
        include_head=True,
        config=eqft.UniformSoupConfig(weights=(1.0, 3.0)),
    )

    assert jnp.allclose(soup.head.weight, tiny_vision_transformer.head.weight + 2.5)


def test_task_vector_rejects_different_base_checkpoint(tiny_vision_transformer):
    tuned = _shift_head(tiny_vision_transformer, 1.0)
    vector = eqft.task_vector(tiny_vision_transformer, tuned, include_head=True)
    other_base = tiny_vision_transformer.__class__(key=jr.PRNGKey(123))

    with pytest.raises(ValueError, match="base checkpoint mismatch"):
        eqft.apply_task_vector(other_base, vector)


def test_task_vector_records_logical_id_lineage(tiny_vision_transformer):
    tuned = _shift_head(tiny_vision_transformer, 1.0)
    vector = eqft.task_vector(tiny_vision_transformer, tuned, include_head=True)
    bundle = eqft.task_vector_bundle(vector, method="task_vector")

    assert vector.logical_id_table_hash
    assert bundle.lineage.logical_id_table_hash == vector.logical_id_table_hash
    assert bundle.lineage.base_value_hash == vector.base_checkpoint_hash

    incompatible = eqft.TaskVector(
        delta=vector.delta,
        include_head=vector.include_head,
        base_architecture_hash=vector.base_architecture_hash,
        base_checkpoint_hash=vector.base_checkpoint_hash,
        logical_id_table_hash="different-logical-id-table",
    )
    with pytest.raises(ValueError, match="logical-ID table mismatch"):
        eqft.apply_task_vector(tiny_vision_transformer, incompatible)


def test_uniform_soup_rejects_declared_base_hash_mismatch():
    model_a = {
        "w": jnp.array([1.0]),
        "metadata": {"base_checkpoint_hash": "base-a"},
    }
    model_b = {
        "w": jnp.array([2.0]),
        "metadata": {"base_checkpoint_hash": "base-b"},
    }

    with pytest.raises(ValueError, match="same base checkpoint"):
        eqft.uniform_soup([model_a, model_b], config=eqft.UniformSoupConfig())


def test_ties_merge_rejects_different_base_checkpoints(tiny_vision_transformer):
    other_base = tiny_vision_transformer.__class__(key=jr.PRNGKey(123))
    vector_a = eqft.task_vector(
        tiny_vision_transformer,
        _shift_head(tiny_vision_transformer, 1.0),
        include_head=True,
    )
    vector_b = eqft.task_vector(
        other_base,
        _shift_head(other_base, 1.0),
        include_head=True,
    )

    with pytest.raises(ValueError, match="same base checkpoint"):
        eqft.ties_merge([vector_a, vector_b])
