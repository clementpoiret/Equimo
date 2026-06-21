"""Regularization helper tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import equimo.finetune as eqft


def test_l2_sp_zero_for_identical_models(tiny_vision_transformer):
    loss = eqft.l2_sp_loss(tiny_vision_transformer, tiny_vision_transformer)

    assert loss == jnp.asarray(0.0)


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
