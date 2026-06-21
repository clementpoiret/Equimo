"""Modern model-merging tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import equimo.finetune as eqft


def _vector(delta, *, logical_id_table_hash: str = "logical-ids"):
    return eqft.TaskVector(
        delta=delta,
        include_head=True,
        base_architecture_hash="arch",
        base_checkpoint_hash="base",
        logical_id_table_hash=logical_id_table_hash,
    )


def test_modern_merges_reject_logical_id_table_mismatch():
    vector_a = _vector({"w": jnp.array([1.0])})
    vector_b = _vector(
        {"w": jnp.array([2.0])},
        logical_id_table_hash="other-logical-ids",
    )

    with pytest.raises(ValueError, match="logical-ID table"):
        eqft.knots_task_vector([vector_a, vector_b])


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


def test_knots_rejects_unsupported_orientation():
    vector = _vector({"w": jnp.eye(2)})

    with pytest.raises(ValueError, match="orientation"):
        eqft.knots_task_vector(
            [vector],
            config=eqft.KnOTSConfig(orientation="in_out"),
        )
