"""PEFT method coverage."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


def test_lora_fa_identity_and_trainable_B_only(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_lora_fa(
        tiny_vision_transformer,
        eqft.LoRAFAConfig(rank=2, alpha=4.0),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="lora_fa",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert plan.trainable.blocks[0].attn.qkv.lora_fa_B is not None
    assert plan.trainable.blocks[0].attn.qkv.frozen_A is None
    assert plan.trainable.blocks[0].attn.qkv.correction_matrix is None
    assert set(plan.report.trainable_by_label) == {"lora_fa_B_decay"}


def test_adalora_orthogonality_aux_loss_spec_is_explicit(tiny_vision_transformer):
    model = eqft.apply_adalora(
        tiny_vision_transformer,
        eqft.AdaLoRAConfig(
            rank=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(11),
    )
    module_loss = model.blocks[0].attn.proj.orthogonality_loss()
    spec = eqft.adalora_orthogonality_aux_loss_spec(coefficient_hint=0.25)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="adalora",
            train_head=False,
        ),
        aux_losses=(spec,),
    )

    assert jnp.allclose(eqft.adalora_orthogonality_loss(model), module_loss)
    assert jnp.allclose(
        eqft.adalora_orthogonality_loss(model, coefficient=0.25),
        module_loss * 0.25,
    )
    assert plan.aux_losses == (spec,)
    assert spec.registry_key == "equimo.adalora_orthogonality_loss"
    assert spec.reduction == "sum"


def test_lora_fa_custom_vjp_freezes_A_and_corrects_B_gradient(tiny_vision_transformer):
    x = jnp.ones((4,), dtype=jnp.float32)
    model = eqft.apply_lora_fa(
        tiny_vision_transformer,
        eqft.LoRAFAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(0),
    )
    module = model.blocks[0].attn.proj

    grads = eqx.filter_grad(lambda m: jnp.sum(m(x)))(module)
    z = module.frozen_A @ x
    raw_grad_B = jnp.outer(jnp.ones((4,), dtype=jnp.float32), z) * module.scaling
    expected_B = (raw_grad_B @ module.correction_matrix) / (module.scaling**2)

    assert jnp.array_equal(grads.frozen_A, jnp.zeros_like(module.frozen_A))
    assert jnp.allclose(grads.lora_fa_B, expected_B, atol=1e-6)


def test_fourierft_identity_trainables_and_merge_equivalence(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_fourierft(
        tiny_vision_transformer,
        eqft.FourierFTConfig(
            num_coefficients=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(1),
    )
    module = model.blocks[0].attn.proj
    trained_module = eqx.tree_at(
        lambda m: m.coefficients_real,
        module,
        jnp.ones_like(module.coefficients_real) * 0.05,
    )
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_fourierft(trained)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="fourierft",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert plan.trainable.blocks[0].attn.proj.coefficients_real is not None
    assert plan.trainable.blocks[0].attn.proj.frequency_indices is None
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_eva_initializes_lora_A_from_activation_artifacts(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    artifacts = {
        "blocks.0.attn.proj": jnp.eye(4, dtype=jnp.float32),
        "blocks.1.attn.proj": jnp.eye(4, dtype=jnp.float32),
    }
    model = eqft.apply_eva_lora(
        tiny_vision_transformer,
        eqft.EVAInitializerConfig(rank_budget=4, per_layer_min_rank=1),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(2),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert model.blocks[0].attn.proj.lora_A.shape[0] >= 1
    assert jnp.array_equal(model.blocks[0].attn.proj.lora_B, jnp.zeros_like(model.blocks[0].attn.proj.lora_B))


def test_eva_accepts_valid_calibration_artifacts(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    artifacts = {
        "blocks.0.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.0.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
        "blocks.1.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.1.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
    }
    model = eqft.apply_eva_lora(
        tiny_vision_transformer,
        eqft.EVAInitializerConfig(
            rank_budget=4,
            per_layer_min_rank=1,
            calibration=eqft.CalibrationSpec(
                artifact_kind="activation_svd",
                sample_count=8,
                data_fingerprint="dataset-a",
            ),
        ),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(22),
    )

    metadata = dict(model.blocks[0].attn.proj.metadata)
    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert metadata["method"] == "eva"
    assert metadata["calibration_sample_count"] == "8"
    assert metadata["calibration_data_fingerprint"] == "dataset-a"


def test_eva_consumes_activation_svd_statistics(tiny_vision_transformer):
    svd_payload = {
        "right_singular_vectors": jnp.eye(4, dtype=jnp.float32),
        "singular_values": jnp.asarray([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32),
    }
    artifacts = {
        "blocks.0.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.0.attn.proj",),
            statistics=svd_payload,
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
        "blocks.1.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.1.attn.proj",),
            statistics=svd_payload,
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
    }

    model = eqft.apply_eva_lora(
        tiny_vision_transformer,
        eqft.EVAInitializerConfig(
            rank_budget=2,
            calibration=eqft.CalibrationSpec(artifact_kind="activation_svd"),
        ),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(25),
    )

    assert model.blocks[0].attn.proj.lora_A.shape == (1, 4)
    assert model.blocks[1].attn.proj.lora_A.shape == (1, 4)


def test_eva_rejects_non_artifact_when_calibration_is_pinned(tiny_vision_transformer):
    artifacts = {
        "blocks.0.attn.proj": jnp.eye(4, dtype=jnp.float32),
        "blocks.1.attn.proj": jnp.eye(4, dtype=jnp.float32),
    }

    with pytest.raises(ValueError, match="CalibrationArtifact"):
        eqft.apply_eva_lora(
            tiny_vision_transformer,
            eqft.EVAInitializerConfig(
                rank_budget=4,
                calibration=eqft.CalibrationSpec(artifact_kind="activation_svd"),
            ),
            activation_artifacts=artifacts,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            key=jr.PRNGKey(23),
        )


def test_eva_rejects_calibration_fingerprint_mismatch(tiny_vision_transformer):
    artifacts = {
        "blocks.0.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.0.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
        "blocks.1.attn.proj": eqft.CalibrationArtifact(
            kind="activation_svd",
            base_checkpoint_hash="base-hash",
            logical_parameter_ids=("blocks.1.attn.proj",),
            statistics=jnp.eye(4, dtype=jnp.float32),
            sample_count=8,
            data_fingerprint="dataset-a",
            accumulation_dtype="float32",
            distributed_reduction="deterministic_sum",
        ),
    }

    with pytest.raises(ValueError, match="data_fingerprint mismatch"):
        eqft.apply_eva_lora(
            tiny_vision_transformer,
            eqft.EVAInitializerConfig(
                rank_budget=4,
                calibration=eqft.CalibrationSpec(
                    artifact_kind="activation_svd",
                    data_fingerprint="dataset-b",
                ),
            ),
            activation_artifacts=artifacts,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
            key=jr.PRNGKey(24),
        )


def test_loftq_initialization_reconstructs_quantized_residual(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    weight = tiny_vision_transformer.blocks[0].attn.proj.weight
    q_weight = weight - jnp.ones_like(weight) * 0.01
    config = eqft.LoftQConfig(
        rank=4,
        quantizer=eqft.QuantizerSpec(id="test", bits=4, format="fake"),
    )
    model = eqft.apply_loftq_lora(
        tiny_vision_transformer,
        config,
        quantized_weights={"blocks.0.attn.proj": q_weight},
        target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        key=jr.PRNGKey(3),
    )
    module = model.blocks[0].attn.proj

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-5)
    assert dict(module.metadata)["method"] == "loftq"
    eqft.validate_loftq_lora(
        model,
        config,
        quantized_weights={"blocks.0.attn.proj": q_weight},
        target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
    )
    with pytest.raises(ValueError, match="quantization fingerprint mismatch"):
        eqft.validate_loftq_lora(
            model,
            config,
            quantized_weights={
                "blocks.0.attn.proj": q_weight + jnp.ones_like(q_weight) * 0.01
            },
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        )


def test_orthogonal_adapter_identity_trainables_and_merge(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_orthogonal_adapters(
        tiny_vision_transformer,
        eqft.OrthogonalAdapterConfig(
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj
    skew = module.skew.at[0, 1].set(0.01)
    trained_module = eqx.tree_at(lambda m: m.skew, module, skew)
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_orthogonal_adapters(trained)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="orthogonal",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert module.orthogonality_error() < 1e-8
    assert plan.trainable.blocks[0].attn.proj.skew is not None
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_boft_blockwise_forward_matches_dense_merge(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_orthogonal_adapters(
        tiny_vision_transformer,
        eqft.OrthogonalAdapterConfig(
            parameterization="butterfly_cayley",
            block_size=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj
    skew = module.skew.at[0, 0, 1].set(0.02)
    trained_module = eqx.tree_at(lambda m: m.skew, module, skew)
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_orthogonal_adapters(trained)

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_boft_adapter_delta_round_trips_with_ordering_metadata(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_orthogonal_adapters(
        tiny_vision_transformer,
        eqft.OrthogonalAdapterConfig(
            parameterization="butterfly_cayley",
            block_size=2,
            num_factors=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj
    skew = module.skew.at[0, 0, 0, 1].set(0.02)
    skew = skew.at[1, 0, 0, 1].set(0.03)
    trained_module = eqx.tree_at(lambda m: m.skew, module, skew)
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_orthogonal_adapters(trained)

    bundle = eqft.extract_adapter_delta(trained)
    loaded = eqft.load_adapter_delta(tiny_vision_transformer, bundle)
    entry = next(
        entry
        for entry in bundle.adapter_config["entries"]
        if entry["class"] == "OrthogonalLinear"
    )
    serialization = entry["orthogonal"]["serialization"]

    assert serialization["layout"] == "sparse_butterfly_cayley"
    assert serialization["block_order"] == (0, 1)
    assert serialization["factor_order"] == (0, 1)
    assert serialization["butterfly_permutation"] == ((0, 1, 2, 3), (0, 2, 1, 3))
    assert jnp.allclose(trained(x), loaded(x), atol=1e-6)
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_boft_requires_explicit_block_size(tiny_vision_transformer):
    with pytest.raises(ValueError, match="block_size"):
        eqft.apply_orthogonal_adapters(
            tiny_vision_transformer,
            eqft.OrthogonalAdapterConfig(
                parameterization="butterfly_cayley",
                target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
            ),
        )


def test_convpass_identity_and_requires_patch_grid(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_convpass(
        tiny_vision_transformer,
        eqft.ConvPassConfig(bottleneck=2, patch_grid=(1, 2)),
        key=jr.PRNGKey(4),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="convpass",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert plan.trainable.blocks[0].convpass.down.weight is not None
    assert plan.trainable.blocks[0].base.attn.proj.weight is None


def test_repadapter_identity_merge_and_delta_round_trip(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_repadapter(
        tiny_vision_transformer,
        eqft.RepAdapterConfig(
            bottleneck=2,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(5),
    )
    module = model.blocks[0].attn.proj
    trained_module = eqx.tree_at(
        lambda m: m.up.weight,
        module,
        jnp.ones_like(module.up.weight) * 0.03,
    )
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_repadapters(trained)
    bundle = eqft.extract_adapter_delta(trained)
    loaded = eqft.load_adapter_delta(tiny_vision_transformer, bundle)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="repadapter",
            train_head=False,
        ),
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert plan.trainable.blocks[0].attn.proj.down.weight is not None
    assert plan.trainable.blocks[0].attn.proj.base.weight is None
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)
    assert jnp.allclose(trained(x), loaded(x), atol=1e-6)


def test_randlora_dense_rank_and_serialized_random_bases(tiny_vision_transformer):
    x = jnp.ones((2, 3))
    model = eqft.apply_randlora(
        tiny_vision_transformer,
        eqft.RandLoRAConfig(
            rank=1,
            basis_count=4,
            alpha=1.0,
            seed=123,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(6),
    )
    module = model.blocks[0].attn.proj
    trained_module = eqx.tree_at(
        lambda m: m.basis_scales,
        module,
        jnp.ones_like(module.basis_scales),
    )
    trained = eqx.tree_at(lambda m: m.blocks[0].attn.proj, model, trained_module)
    merged = eqft.merge_randlora(trained)
    bundle = eqft.extract_lora_delta(trained)
    loaded = eqft.load_lora_delta(tiny_vision_transformer, bundle)
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="randlora",
            train_head=False,
        ),
    )
    entry = next(
        entry
        for entry in bundle.adapter_config["entries"]
        if entry["class"] == "RandLoRALinear"
    )

    assert jnp.allclose(tiny_vision_transformer(x), model(x), atol=1e-6)
    assert plan.trainable.blocks[0].attn.proj.basis_scales is not None
    assert plan.trainable.blocks[0].attn.proj.random_A is None
    assert plan.trainable.blocks[0].attn.proj.random_B is None
    assert int(jnp.linalg.matrix_rank(trained_module.delta_weight())) > 1
    assert entry["seed"] == 123
    assert entry["random_A"].shape == (4, 1, 4)
    assert entry["random_B"].shape == (4, 4, 1)
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)
    assert jnp.allclose(trained(x), loaded(x), atol=1e-6)


def test_profile_registry_declares_required_methods_and_artifacts():
    ids = set(eqft.available_profile_ids())

    assert "lora_fa.zhang2026.corrected_v3" in ids
    assert "eva.initializer.reference" in ids
    assert "loftq.initializer.reference" in ids
    assert "fourierft.gao2024.reference" in ids
    assert "boft.liu2024.butterfly_cayley" in ids
    assert "convpass.vision.reference" in ids
    assert "repadapter.luo2023.structural" in ids
    assert "randlora.albert2025.reference" in ids

    lora_fa = eqft.lora_fa_zhang2026_profile()
    historical = eqft.lora_fa_zhang2026_profile(
        eqft.LoRAFAConfig(gradient_mode="frozen_A", custom_vjp=False)
    )
    eva = eqft.eva_initializer_profile(
        eqft.EVAInitializerConfig(
            rank_budget=4,
            calibration=eqft.CalibrationSpec(
                artifact_kind="activation_svd",
                sample_count=8,
            ),
        )
    )
    loftq = eqft.loftq_initializer_profile(
        eqft.LoftQConfig(
            rank=2,
            quantizer=eqft.QuantizerSpec(id="nf4", bits=4, format="nf4"),
        )
    )
    convpass = eqft.convpass_profile(eqft.ConvPassConfig(patch_grid=None))
    repadapter = eqft.repadapter_profile()
    randlora = eqft.randlora_profile(eqft.RandLoRAConfig(seed=None))

    assert lora_fa.fidelity == "reference_implementation"
    assert historical.fidelity == "experimental"
    assert historical.known_deviations
    assert eva.required_artifacts == ("activation_svd",)
    assert loftq.required_artifacts == ("quantized_base", "quantization_residual")
    assert convpass.required_artifacts == ("patch_grid_metadata",)
    assert repadapter.fidelity == "reference_implementation"
    assert randlora.fidelity == "experimental"
    assert "explicit seed" in randlora.known_deviations[0]
    assert lora_fa.target_spec["tags_any"] == ("attention.qkv", "attention.proj")


def test_profiles_do_not_claim_exactness_for_unpinned_random_choices():
    profile = eqft.fourierft_gao2024_profile(
        eqft.FourierFTConfig(num_coefficients=2, frequency_selection="random")
    )
    boft = eqft.boft_liu2024_profile(
        eqft.OrthogonalAdapterConfig(parameterization="cayley")
    )

    assert profile.fidelity == "reference_implementation"
    assert "explicit seed" in profile.known_deviations[0]
    assert boft.fidelity == "experimental"
    assert boft.known_deviations
