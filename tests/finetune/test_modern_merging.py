"""Modern model-merging tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import equimo.finetune as eqft


def _vector(delta):
    return eqft.TaskVector(
        delta=delta,
        include_head=True,
        base_architecture_hash="arch",
        base_checkpoint_hash="base",
    )


def test_adamerging_coefficients_and_bundle_metadata():
    vector_a = _vector({"w": jnp.array([[1.0, 0.0], [0.0, 0.0]])})
    vector_b = _vector({"w": jnp.array([[3.0, 0.0], [0.0, 0.0]])})
    config = eqft.AdaMergingConfig(
        coefficients=(0.25, 0.75),
        data_fingerprint="unlabeled:v1",
        coefficient_source="learned",
    )

    merged = eqft.adamerging_task_vector([vector_a, vector_b], config=config)

    assert jnp.allclose(merged.delta["w"], jnp.array([[2.5, 0.0], [0.0, 0.0]]))
    assert merged.metadata["data_fingerprint"] == "unlabeled:v1"
    method = eqft.AdaMerging(config)
    bundle = method.merge(method.prepare([vector_a, vector_b]))
    assert bundle.method == "adamerging"
    assert bundle.delta_tree["w"].shape == (2, 2)
    assert bundle.lineage.base_checkpoint_hash == "base"
    assert bundle.metadata["coefficients"] == pytest.approx((0.25, 0.75))


def test_adamerging_layer_coefficients_override_defaults():
    vector_a = _vector(
        {
            "w": jnp.array([1.0]),
            "b": jnp.array([10.0]),
        }
    )
    vector_b = _vector(
        {
            "w": jnp.array([3.0]),
            "b": jnp.array([30.0]),
        }
    )
    config = eqft.AdaMergingConfig(
        coefficients=(0.5, 0.5),
        layer_coefficients={"b": (0.0, 1.0)},
    )

    merged = eqft.adamerging_task_vector([vector_a, vector_b], config=config)

    assert jnp.allclose(merged.delta["w"], jnp.array([2.0]))
    assert jnp.allclose(merged.delta["b"], jnp.array([30.0]))


def test_adamerging_learned_coefficients_require_fingerprint():
    vector = _vector({"w": jnp.array([1.0])})

    with pytest.raises(ValueError, match="data_fingerprint"):
        eqft.adamerging_task_vector(
            [vector],
            config=eqft.AdaMergingConfig(
                coefficients=(1.0,),
                coefficient_source="learned",
            ),
        )


def test_learn_adamerging_coefficients_records_unlabeled_data_fingerprint():
    target = jnp.array([0.1, 0.9], dtype=jnp.float32)

    def loss_fn(coefficients, batch):
        return jnp.sum(jnp.square(coefficients - target)) + batch * 0.0

    config = eqft.learn_adamerging_coefficients(
        loss_fn,
        (jnp.array(1.0), jnp.array(2.0)),
        task_count=2,
        data_fingerprint="unlabeled:v2",
        steps=80,
        learning_rate=0.5,
    )

    assert config.coefficient_source == "learned"
    assert config.data_fingerprint == "unlabeled:v2"
    assert config.coefficients[1] > 0.8
    assert sum(config.coefficients) == pytest.approx(1.0)


def test_knots_shared_left_basis_merge_and_metadata():
    vector_a = _vector({"w": jnp.array([[1.0, 0.0], [0.0, 0.0]])})
    vector_b = _vector({"w": jnp.array([[2.0, 0.0], [0.0, 0.0]])})

    merged = eqft.knots_task_vector(
        [vector_a, vector_b],
        config=eqft.KnOTSConfig(rank=1),
    )

    assert jnp.allclose(merged.delta["w"], jnp.array([[1.5, 0.0], [0.0, 0.0]]))
    assert merged.metadata["basis_policy"] == "shared_left_svd"
    assert merged.metadata["orientation"] == "out_in"


def test_tsv_shared_right_basis_merge_and_bundle():
    vector_a = _vector({"w": jnp.array([[1.0, 0.0], [0.0, 0.0]])})
    vector_b = _vector({"w": jnp.array([[0.0, 1.0], [0.0, 0.0]])})
    method = eqft.TSVMerging(eqft.TSVConfig(rank=1))

    plan = method.prepare([vector_a, vector_b])
    bundle = method.merge(plan)

    assert jnp.allclose(bundle.delta_tree["w"], jnp.array([[0.5, 0.5], [0.0, 0.0]]))
    assert bundle.metadata["orthogonalization_policy"] == (
        "shared_right_svd_orthogonal_basis"
    )
    assert bundle.adapter_config["method_config"]["rank"] == 1
