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

    assert plan.trainable.blocks[0].mlp.adapters[0].down.weight is not None
    assert plan.trainable.blocks[0].mlp.base.fc1.weight is None
    assert plan.trainable.head.weight is not None
    assert "adapter_decay" in plan.report.trainable_by_label


def test_adapter_train_base_includes_wrapped_base_leaves(tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="after_mlp", train_base=True),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        adapted,
        trainable=eqft.TrainableSpec(mode="peft", method_name="adapter"),
    )

    assert adapted.blocks[0].mlp.train_base
    assert plan.trainable.blocks[0].mlp.adapters[0].down.weight is not None
    assert plan.trainable.blocks[0].mlp.base.fc1.weight is not None


def test_adapter_both_placement(tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="both"),
        key=jr.PRNGKey(0),
    )

    assert isinstance(adapted.blocks[0].attn, eqft.OutputAdapterModule)
    assert isinstance(adapted.blocks[0].mlp, eqft.OutputAdapterModule)
    assert len(adapted.blocks[0].attn.adapters) == 1
    assert len(adapted.blocks[0].mlp.adapters) == 1
    assert len(adapted.blocks[1].attn.adapters) == 1
    assert len(adapted.blocks[1].mlp.adapters) == 1


def test_adapter_config_spec_fields_and_residual_scale(tiny_vision_transformer):
    config = eqft.AdapterConfig(
        bottleneck=3,
        residual_scale_init=0.5,
        pre_norm=False,
        train_base=False,
    )
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        config,
        key=jr.PRNGKey(0),
    )

    assert config.residual_scale_init == 0.5
    assert adapted.blocks[0].mlp.adapters[0].residual_scale == jnp.asarray(0.5)


def test_adapter_pre_norm_inserts_trainable_adapter_norm(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="after_mlp", pre_norm=True),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        adapted,
        trainable=eqft.TrainableSpec(mode="peft", method_name="adapter"),
    )

    assert isinstance(adapted.blocks[0].mlp.adapters[0].norm, eqx.nn.LayerNorm)
    assert jnp.allclose(tiny_vision_transformer(x), adapted(x), atol=1e-6)
    assert plan.trainable.blocks[0].mlp.adapters[0].norm.weight is not None
    assert plan.labels.blocks[0].mlp.adapters[0].norm.weight == "adapter_no_decay"


def test_parallel_adapter_config_uses_parallel_wrapper(tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.ParallelAdapterConfig(bottleneck=3),
        key=jr.PRNGKey(0),
    )

    assert isinstance(adapted.blocks[0], eqft.ParallelAdapterBlock)
    assert adapted.blocks[0].adapter.down.out_features == 3


def test_parallel_adapter_train_base_includes_wrapped_base_leaves(
    tiny_vision_transformer,
):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.ParallelAdapterConfig(bottleneck=3, train_base=True),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        adapted,
        trainable=eqft.TrainableSpec(mode="peft", method_name="adapter"),
    )

    assert adapted.blocks[0].train_base
    assert plan.trainable.blocks[0].adapter.down.weight is not None
    assert plan.trainable.blocks[0].base.attn.proj.weight is not None


def test_adapter_delta_roundtrip(tmp_path, tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="after_mlp"),
        key=jr.PRNGKey(0),
    )
    adapter = adapted.blocks[0].mlp.adapters[0]
    adapter = eqx.tree_at(
        lambda m: m.up.weight,
        adapter,
        jnp.ones_like(adapter.up.weight) * 0.01,
    )
    adapted = eqx.tree_at(lambda m: m.blocks[0].mlp.adapters[0], adapted, adapter)
    x = jnp.ones((2, 3))
    path = tmp_path / "adapter.eqft"

    eqft.save_delta(adapted, path, method="adapter")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.allclose(adapted(x), loaded(x), atol=1e-6)
    assert isinstance(loaded.blocks[0].mlp.adapters[0].down, eqx.nn.Linear)
    assert jnp.array_equal(loaded.blocks[0].mlp.adapters[0].up.weight, adapter.up.weight)
    assert jnp.array_equal(
        loaded.blocks[0].mlp.adapters[0].residual_scale,
        adapter.residual_scale,
    )


def test_adapter_pre_norm_delta_roundtrip(tmp_path, tiny_vision_transformer):
    adapted = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3, placement="after_mlp", pre_norm=True),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "adapter_pre_norm.eqft"

    eqft.save_delta(adapted, path, method="adapter")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert isinstance(loaded.blocks[0].mlp.adapters[0].norm, eqx.nn.LayerNorm)
    assert jnp.array_equal(
        loaded.blocks[0].mlp.adapters[0].norm.weight,
        adapted.blocks[0].mlp.adapters[0].norm.weight,
    )


def test_adaptformer_parallel_mlp_identity(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    adapted = eqft.apply_adaptformer(
        tiny_vision_transformer,
        eqft.AdaptFormerConfig(bottleneck=3),
        key=jr.PRNGKey(0),
    )

    assert isinstance(adapted.blocks[0], eqft.AdaptFormerBlock)
    assert jnp.allclose(tiny_vision_transformer(x), adapted(x), atol=1e-6)
    plan = eqft.prepare_finetune(
        adapted,
        trainable=eqft.TrainableSpec(mode="peft", method_name="adaptformer"),
    )
    assert plan.trainable.blocks[0].adapter.scale is not None


def test_adaptformer_paper_profile_freezes_fixed_scale(tiny_vision_transformer):
    adapted = eqft.apply_adaptformer(
        tiny_vision_transformer,
        eqft.AdaptFormerConfig.paper_chen2022(),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        adapted,
        trainable=eqft.TrainableSpec(mode="peft", method_name="adaptformer"),
    )

    assert adapted.blocks[0].adapter.adapter.activation == "relu"
    assert adapted.blocks[0].adapter.scale_trainable is False
    assert jnp.allclose(adapted.blocks[0].adapter.scale, jnp.asarray(0.1))
    assert plan.trainable.blocks[0].adapter.scale is None


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

    assert model.blocks[0].mlp.adapter_names == ("dataset_a", "dataset_b")
    assert model.blocks[0].mlp.active_adapter == "dataset_b"
    with pytest.raises(ValueError, match="not found"):
        eqft.set_active_adapter(model, "missing")


def test_adapter_bank_config_supports_policy_and_multiple_active(tiny_vision_transformer):
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

    with pytest.raises(ValueError, match="Multiple active"):
        eqft.configure_adapter_bank(
            model,
            eqft.AdapterBankConfig(active=("dataset_a", "dataset_b")),
        )

    configured = eqft.configure_adapter_bank(
        model,
        eqft.AdapterBankConfig(
            active=("dataset_a", "dataset_b"),
            allow_multiple_active=True,
        ),
    )
    ignored = eqft.configure_adapter_bank(
        model,
        eqft.AdapterBankConfig(active="missing", missing_adapter_policy="ignore"),
    )

    assert configured.blocks[0].mlp.active_adapter == ("dataset_a", "dataset_b")
    assert ignored.blocks[0].mlp.active_adapter == "dataset_a"


def test_adapter_fusion_attaches_attention_to_named_banks(tiny_vision_transformer):
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
    fused = eqft.apply_adapter_fusion(
        model,
        eqft.AdapterFusionConfig(fusion_dropout=0.0),
        key=jr.PRNGKey(2),
    )
    adapter_outputs = (jnp.zeros((4,)), jnp.zeros((4,)))
    weights = fused.blocks[0].mlp.adapter_fusion.attention_weights(
        jnp.ones((4,)),
        adapter_outputs,
    )

    assert isinstance(fused.blocks[0].mlp.adapter_fusion, eqft.AdapterFusion)
    assert fused.blocks[0].mlp.active_adapter == ("dataset_a", "dataset_b")
    assert fused.blocks[0].mlp.adapter_fusion.query.in_features == 4
    assert fused.blocks[0].mlp.adapter_fusion.key.in_features == 4
    assert fused.blocks[0].mlp.adapter_fusion.value.in_features == 4
    assert jnp.allclose(weights, jnp.array([0.5, 0.5]))
    assert fused(jnp.ones((2, 3))).shape == (2,)


def test_adapter_fusion_attention_uses_adapter_keys():
    fusion = eqft.AdapterFusion(2, 2, key=jr.PRNGKey(0))
    fusion = eqx.tree_at(
        lambda module: (
            module.query.weight,
            module.key.weight,
            module.value.weight,
        ),
        fusion,
        (
            jnp.eye(2, dtype=jnp.float32),
            jnp.eye(2, dtype=jnp.float32),
            jnp.eye(2, dtype=jnp.float32),
        ),
    )
    weights = fusion.attention_weights(
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        (
            jnp.asarray([2.0, 0.0], dtype=jnp.float32),
            jnp.asarray([-2.0, 0.0], dtype=jnp.float32),
        ),
    )

    assert weights[0] > weights[1]


def test_adapter_fusion_respects_configured_placement(tiny_vision_transformer):
    model = eqft.add_adapter(
        tiny_vision_transformer,
        name="dataset_a",
        config=eqft.AdapterConfig(bottleneck=3, placement="both"),
        key=jr.PRNGKey(0),
    )
    model = eqft.add_adapter(
        model,
        name="dataset_b",
        config=eqft.AdapterConfig(bottleneck=3, placement="both"),
        key=jr.PRNGKey(1),
    )

    fused = eqft.apply_adapter_fusion(
        model,
        eqft.AdapterFusionConfig(placement=("after_mlp",)),
        key=jr.PRNGKey(2),
    )

    assert fused.blocks[0].attn.adapter_fusion is None
    assert isinstance(fused.blocks[0].mlp.adapter_fusion, eqft.AdapterFusion)


def test_adapter_fusion_trainable_spec_freezes_task_adapters(tiny_vision_transformer):
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
    fused = eqft.apply_adapter_fusion(model, key=jr.PRNGKey(2))
    plan = eqft.prepare_finetune(
        fused,
        trainable=eqft.adapter_fusion_trainable_spec(train_head=False),
    )

    assert plan.trainable.blocks[0].mlp.adapter_fusion.query.weight is not None
    assert plan.trainable.blocks[0].mlp.adapter_fusion.key.weight is not None
    assert plan.trainable.blocks[0].mlp.adapter_fusion.value.weight is not None
    assert plan.trainable.blocks[0].mlp.adapters[0].down.weight is None
    assert plan.trainable.head.weight is None


def test_adapter_fusion_delta_roundtrip(tmp_path, tiny_vision_transformer):
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
    fused = eqft.apply_adapter_fusion(model, key=jr.PRNGKey(2))
    trained = eqx.tree_at(
        lambda m: m.blocks[0].mlp.adapter_fusion.query.weight,
        fused,
        jnp.ones_like(fused.blocks[0].mlp.adapter_fusion.query.weight) * 0.1,
    )
    path = tmp_path / "adapter_fusion.eqft"

    eqft.save_delta(trained, path, method="adapter")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert isinstance(loaded.blocks[0].mlp.adapter_fusion, eqft.AdapterFusion)
    assert jnp.array_equal(
        loaded.blocks[0].mlp.adapter_fusion.query.weight,
        trained.blocks[0].mlp.adapter_fusion.query.weight,
    )


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
