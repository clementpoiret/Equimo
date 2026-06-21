"""Bottleneck adapter tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


def test_adapter_zero_up_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(tiny_vision_transformer(x), adapted(x), atol=1e-6)


def test_adapter_after_mlp_trainable_leaves(tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        adapted,
        trainable=eqft.TrainableSpec(mode="peft", method_name="adapter", train_head=True),
    )

    assert plan.trainable.blocks[0].adapters[0].down.weight is not None
    assert plan.trainable.blocks[0].base.attn.qkv.weight is None
    assert plan.trainable.head.weight is not None
    assert "adapter_decay" in plan.report.trainable_by_label


def test_adapter_both_placement(tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="both"),
        key=jr.PRNGKey(0),
    )

    assert isinstance(adapted.blocks[0], eqft.SerialAdapterBlock)
    assert len(adapted.blocks[0].adapters) == 2
    assert len(adapted.blocks[1].adapters) == 2


def test_adapter_delta_roundtrip(tmp_path, tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
        key=jr.PRNGKey(0),
    )
    adapter = adapted.blocks[0].adapters[0]
    adapter = eqx.tree_at(
        lambda m: m.up.weight,
        adapter,
        jnp.ones_like(adapter.up.weight) * 0.01,
    )
    adapted = eqx.tree_at(lambda m: m.blocks[0].adapters[0], adapted, adapter)
    x = jnp.ones((2, 3))
    path = tmp_path / "adapter.eqft"

    eqft.save_delta(adapted, path, method="adapter")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.allclose(adapted(x), loaded(x), atol=1e-6)
    assert isinstance(loaded.blocks[0].adapters[0].down, eqx.nn.Linear)
    assert jnp.array_equal(loaded.blocks[0].adapters[0].up.weight, adapter.up.weight)


def test_adaptformer_parallel_mlp_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    adapted = eqft.apply_adaptformer(
        tiny_vision_transformer,
        eqft.AdaptFormerConfig(bottleneck=3),
        key=jr.PRNGKey(0),
    )

    assert isinstance(adapted.blocks[0], eqft.AdaptFormerBlock)
    assert jnp.allclose(tiny_vision_transformer(x), adapted(x), atol=1e-6)


def test_named_adapter_bank_switches_active_adapter(tiny_vision_transformer):
    model = eqft.add_adapter(
        tiny_vision_transformer,
        name="dataset_a",
        config=eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
        key=jr.PRNGKey(0),
    )
    model = eqft.add_adapter(
        model,
        name="dataset_b",
        config=eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
        key=jr.PRNGKey(1),
    )
    model = eqft.set_active_adapter(model, "dataset_b")

    assert model.blocks[0].adapter_names == ("dataset_a", "dataset_b")
    assert model.blocks[0].active_adapter == "dataset_b"
    with pytest.raises(ValueError, match="not found"):
        eqft.set_active_adapter(model, "missing")


def test_named_adapter_bank_rejects_duplicate_and_parallel(tiny_vision_transformer):
    model = eqft.add_adapter(
        tiny_vision_transformer,
        name="dataset_a",
        config=eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
        key=jr.PRNGKey(0),
    )

    with pytest.raises(ValueError, match="already exists"):
        eqft.add_adapter(
            model,
            name="dataset_a",
            config=eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
            key=jr.PRNGKey(1),
        )
    with pytest.raises(ValueError, match="support"):
        eqft.add_adapter(
            tiny_vision_transformer,
            name="parallel",
            config=eqft.AdapterConfig(bottleneck=3, placement="parallel"),
            key=jr.PRNGKey(2),
        )
