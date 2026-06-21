"""Regularization helper tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import equinox as eqx

import equimo.finetune as eqft


def test_l2_sp_zero_for_identical_models(tiny_vision_transformer):
    loss = eqft.l2_sp_loss(tiny_vision_transformer, tiny_vision_transformer)

    assert loss == jnp.asarray(0.0)


def test_l2_sp_reduction_and_explicit_coefficient():
    model = jnp.array([1.0, 3.0])
    reference = jnp.array([0.0, 1.0])

    unscaled = eqft.l2_sp_loss(model, reference)
    scaled_sum = eqft.l2_sp_loss(
        model,
        reference,
        coefficient=0.5,
        reduction="sum",
    )

    assert jnp.allclose(unscaled, jnp.asarray(0.0025))
    assert scaled_sum == jnp.asarray(2.5)


def test_l2_sp_respects_default_excluded_tags(tiny_vision_transformer):
    changed_head = tiny_vision_transformer.head
    changed_head = eqx.tree_at(
        lambda head: head.weight,
        changed_head,
        changed_head.weight + 10.0,
    )
    model = eqx.tree_at(lambda m: m.head, tiny_vision_transformer, changed_head)

    excluded = eqft.l2_sp_loss(model, tiny_vision_transformer)
    included = eqft.l2_sp_loss(
        model,
        tiny_vision_transformer,
        config=eqft.L2SPConfig(shared_mask="all"),
    )

    assert excluded == jnp.asarray(0.0)
    assert included > 0.0


def test_adapter_and_task_vector_norm_penalties(tiny_vision_transformer):
    lora = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jax.random.PRNGKey(0),
    )
    tuned_head = eqx.tree_at(
        lambda m: m.head.weight,
        tiny_vision_transformer,
        tiny_vision_transformer.head.weight + 1.0,
    )
    vector = eqft.task_vector(tiny_vision_transformer, tuned_head, include_head=True)

    assert eqft.adapter_norm_loss(lora) > 0.0
    assert eqft.task_vector_norm_loss(vector, reduction="sum") == jnp.asarray(
        tiny_vision_transformer.head.weight.size,
        dtype=jnp.float32,
    )


def test_ewc_loss_uses_supplied_fisher_diagonal():
    model = jnp.array([1.0, 3.0])
    anchor = jnp.array([0.0, 1.0])
    fisher = jnp.array([2.0, 0.5])

    loss = eqft.ewc_loss(model, anchor, fisher)
    scaled_mean = eqft.ewc_loss(
        model,
        anchor,
        fisher,
        coefficient=0.5,
        reduction="mean",
    )

    assert loss == jnp.asarray(4.0)
    assert scaled_mean == jnp.asarray(1.0)


def test_ewc_config_rejects_non_diagonal_fisher():
    with pytest.raises(ValueError, match="diagonal"):
        eqft.ewc_loss(
            jnp.array([1.0]),
            jnp.array([0.0]),
            jnp.array([1.0]),
            config=eqft.EWCConfig(fisher="full"),  # type: ignore[arg-type]
        )


def test_mixout_config_matches_metadata_defaults():
    config = eqft.MixoutConfig()

    assert config.p == 0.1
    assert config.anchor == "pretrained"
    assert config.target == "full_or_partial_trainable_weights"


def test_mixout_leaf_matches_anchor_preserving_formula():
    key = jax.random.PRNGKey(0)
    leaf = jnp.array([1.0, 3.0, 5.0])
    anchor = jnp.array([0.0, 1.0, 2.0])
    keep_prob = 0.5
    mask = jax.random.bernoulli(key, keep_prob, leaf.shape)

    mixed = eqft.mixout_leaf(leaf, anchor, key=key, p=0.5)

    assert jnp.array_equal(
        mixed,
        anchor + jnp.where(mask, (leaf - anchor) / keep_prob, 0),
    )


def test_mixout_tree_inference_and_probability_validation():
    tree = {"weight": jnp.array([1.0, 2.0]), "name": "leaf"}
    anchor = {"weight": jnp.array([0.0, 0.0]), "name": "anchor"}

    mixed = eqft.mixout_tree(
        tree,
        anchor,
        key=jax.random.PRNGKey(0),
        p=0.5,
        inference=True,
    )

    assert jnp.array_equal(mixed["weight"], tree["weight"])
    assert mixed["name"] == "leaf"
    with pytest.raises(ValueError, match="0 <= p < 1"):
        eqft.mixout_leaf(
            jnp.array([1.0]),
            jnp.array([0.0]),
            key=jax.random.PRNGKey(0),
            p=1.0,
        )


def test_feature_distillation_stops_teacher_gradient():
    student = jnp.array([[1.0, 2.0]])
    teacher = jnp.array([[2.0, 4.0]])

    grad_teacher = jax.grad(
        lambda teacher_features: eqft.feature_distillation_loss(
            student,
            teacher_features,
        )
    )(teacher)

    assert jnp.array_equal(grad_teacher, jnp.zeros_like(teacher))


def test_feature_distillation_dense_preset_and_scaling():
    config = eqft.FeatureDistillationConfig.dense(normalize_features=False)
    student = jnp.array([[1.0, 2.0]])
    teacher = jnp.array([[2.0, 4.0]])

    loss = eqft.feature_distillation_loss(
        student,
        teacher,
        config=config,
        coefficient=0.5,
    )

    assert config.layers == ("25%", "50%", "75%", "100%")
    assert loss == jnp.asarray(1.25)


def test_select_feature_taps_from_mapping_by_name_and_percentage():
    taps = {
        "early": jnp.array([1.0]),
        "middle": jnp.array([2.0]),
        "late": jnp.array([3.0]),
    }

    selected = eqft.select_feature_taps(taps, ("middle", "100%"))

    assert jnp.array_equal(selected[0], jnp.array([2.0]))
    assert jnp.array_equal(selected[1], jnp.array([3.0]))


def test_select_feature_taps_from_sequence_by_index_and_percentage():
    taps = (
        jnp.array([1.0]),
        jnp.array([2.0]),
        jnp.array([3.0]),
    )

    selected = eqft.select_feature_taps(taps, ("1", "100%"))

    assert jnp.array_equal(selected[0], jnp.array([2.0]))
    assert jnp.array_equal(selected[1], jnp.array([3.0]))


def test_feature_distillation_from_taps_stops_teacher_gradient():
    student = {
        "early": jnp.array([[1.0, 2.0]]),
        "late": jnp.array([[2.0, 4.0]]),
    }
    config = eqft.FeatureDistillationConfig(
        layers=("early", "late"),
        normalize_features=False,
    )

    grad_early, grad_late = jax.grad(
        lambda early, late: eqft.feature_distillation_loss_from_taps(
            student,
            {"early": early, "late": late},
            config=config,
        ),
        argnums=(0, 1),
    )(
        jnp.array([[2.0, 4.0]]),
        jnp.array([[4.0, 8.0]]),
    )

    assert jnp.array_equal(grad_early, jnp.zeros_like(grad_early))
    assert jnp.array_equal(grad_late, jnp.zeros_like(grad_late))


def test_feature_distillation_from_taps_aggregates_selected_layers():
    config = eqft.FeatureDistillationConfig(
        layers=("early", "late"),
        normalize_features=False,
    )
    loss = eqft.feature_distillation_loss_from_taps(
        {
            "early": jnp.array([0.0]),
            "late": jnp.array([0.0]),
        },
        {
            "early": jnp.array([1.0]),
            "late": jnp.array([3.0]),
        },
        config=config,
    )

    assert loss == jnp.asarray(5.0)


def test_select_feature_taps_errors_are_clear():
    with pytest.raises(ValueError, match="empty"):
        eqft.select_feature_taps({}, ("50%",))
    with pytest.raises(ValueError, match="requires named taps"):
        eqft.select_feature_taps((jnp.array([1.0]),), ("late",))
    with pytest.raises(ValueError, match="Percentage"):
        eqft.select_feature_taps({"early": jnp.array([1.0])}, ("0%",))
    with pytest.raises(ValueError, match="not found"):
        eqft.select_feature_taps({"early": jnp.array([1.0])}, ("late",))
