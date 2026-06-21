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


def test_adalora_rank_pattern_zeros_singulars_without_persistent_mask(
    tiny_vision_transformer,
):
    model = eqft.apply_adalora(
        tiny_vision_transformer,
        eqft.AdaLoRAConfig(
            rank=3,
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
        key=jr.PRNGKey(12),
    )
    path, module = eqft.iter_adalora_modules(model)[0]
    singular = jnp.asarray([1.0, 2.0, 3.0], dtype=module.singular.dtype)
    model = eqx.tree_at(
        lambda tree: eqft.iter_adalora_modules(tree)[0][1].singular,
        model,
        singular,
    )
    rank_groups = eqft.lora_rank_groups(model)
    name = eqft.path_to_str(path)

    masked = eqft.apply_lora_rank_pattern(
        model,
        {name: jnp.asarray([True, False, True])},
    )
    masked_module = eqft.iter_adalora_modules(masked)[0][1]

    assert rank_groups == {name: 3}
    assert masked_module.final_mask is None
    assert jnp.allclose(masked_module.singular, jnp.asarray([1.0, 0.0, 3.0]))

    final = eqft.apply_lora_rank_pattern(
        model,
        {name: jnp.asarray([False, True, True])},
        final=True,
    )
    final_module = eqft.iter_adalora_modules(final)[0][1]

    assert jnp.array_equal(final_module.final_mask, jnp.asarray([False, True, True]))
    assert jnp.allclose(final_module.singular, jnp.asarray([0.0, 2.0, 3.0]))


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
    assert module.frequency_selection == "random"
    assert (
        module.transform_normalization
        == "jax.numpy.fft.ifft default 1/n inverse scaling"
    )
    assert module.reshape_convention == "row_major_flatten_out_in"
    assert plan.trainable.blocks[0].attn.proj.coefficients_real is not None
    assert plan.trainable.blocks[0].attn.proj.frequency_indices is None
    assert jnp.allclose(trained(x), merged(x), atol=1e-6)


def test_fourierft_deduplicates_conjugate_frequencies(tiny_vision_transformer):
    model = eqft.apply_fourierft(
        tiny_vision_transformer,
        eqft.FourierFTConfig(
            num_coefficients=2,
            frequency_selection="explicit",
            frequency_indices=(1, -1),
            target=eqft.TargetSpec(tags_any=("attention.proj",), max_depth=0),
        ),
    )
    module = model.blocks[0].attn.proj

    assert module.frequency_indices.shape == (1,)
    assert module.frequency_selection == "explicit"
    assert (
        module.delta_weight().dtype
        == tiny_vision_transformer.blocks[0].attn.proj.weight.dtype
    )


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
    assert jnp.array_equal(
        model.blocks[0].attn.proj.lora_B,
        jnp.zeros_like(model.blocks[0].attn.proj.lora_B),
    )


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


def test_eva_rank_allocation_ties_use_logical_parameter_id(tiny_vision_transformer):
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
            rank_budget=3,
            calibration=eqft.CalibrationSpec(artifact_kind="activation_svd"),
        ),
        activation_artifacts=artifacts,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(26),
    )

    assert model.blocks[0].attn.proj.lora_A.shape == (2, 4)
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
    assert "fourierft.gao2024.reference" in ids
    assert "boft.liu2024.butterfly_cayley" in ids
    assert "randlora.albert2025.reference" in ids
    assert "lora.equimo_default" in ids
    assert "lora.hu2021.qv" in ids
    assert "rslora.kalajdzievski2023" in ids
    assert "dora.liu2024.paper_equation" in ids
    assert "dora.nvlabs_reference" in ids
    assert "dora.factored_2026.eager" in ids
    assert "pissa.meng2024.exact_svd" in ids
    assert "vera.kopiczko2024.shape_compatible" in ids
    assert "adapter.houlsby2019.bottleneck" in ids
    assert "adapterfusion.pfeiffer2021.attention" in ids
    assert "adaptformer.chen2022.paper" in ids
    assert "vpt.jia2022.deep" in ids
    assert "vpt.jia2022.shallow" in ids
    assert "soft_prompt.lester2021.input" in ids
    assert "ptuning_v2.liu2022.deep_prompts" in ids
    assert "prefix_tuning.li2021.attention_kv" in ids
    assert "ia3.liu2022.activation_scaling" in ids
    assert "ssf.lian2022.vit" in ids

    houlsby = eqft.adapter_houlsby2019_profile()
    single_site_adapter = eqft.adapter_houlsby2019_profile(
        eqft.AdapterConfig(placement="after_mlp")
    )
    adapterfusion = eqft.adapterfusion_pfeiffer2021_profile()
    unfrozen_fusion = eqft.adapterfusion_pfeiffer2021_profile(
        eqft.AdapterFusionConfig(freeze_task_adapters=False)
    )
    adaptformer = eqft.adaptformer_chen2022_profile()
    equimo_adaptformer = eqft.adaptformer_chen2022_profile(
        eqft.AdaptFormerConfig.safe_default()
    )
    vpt_deep = eqft.vpt_deep_jia2022_profile()
    vpt_wrong_insert = eqft.vpt_deep_jia2022_profile(
        eqft.VPTDeepConfig(prepend_to="before_all")
    )
    vpt_shallow = eqft.vpt_shallow_jia2022_profile()
    soft_prompt = eqft.soft_prompt_lester2021_profile()
    deep_soft_prompt = eqft.soft_prompt_lester2021_profile(
        eqft.SoftPromptConfig(depth="deep")
    )
    ptuning_v2 = eqft.ptuning_v2_liu2022_profile()
    ptuning_shared = eqft.ptuning_v2_liu2022_profile(
        eqft.PTuningV2Config(share_across_layers=True)
    )
    prefix_tuning = eqft.prefix_tuning_li2021_profile()
    prefix_no_projection = eqft.prefix_tuning_li2021_profile(
        eqft.PrefixConfig(prefix_projection=False)
    )
    ia3 = eqft.ia3_liu2022_profile()
    ia3_missing_value = eqft.ia3_liu2022_profile(
        eqft.IA3Config(target=eqft.TargetSpec(tags_any=("attention.k", "mlp.hidden")))
    )
    ssf = eqft.ssf_lian2022_vit_profile()
    broad_ssf = eqft.ssf_lian2022_vit_profile(eqft.ScaleShiftConfig())
    lora_default = eqft.lora_equimo_default_profile()
    lora_qv = eqft.lora_hu2021_qv_profile()
    rslora = eqft.rslora_kalajdzievski2023_profile()
    ordinary_scaled_rslora = eqft.rslora_kalajdzievski2023_profile(
        eqft.RsLoRAConfig(scaling="alpha_over_r")
    )
    dora_paper = eqft.dora_liu2024_profile()
    dora_reference = eqft.dora_nvlabs_reference_profile()
    full_gradient_reference = eqft.dora_nvlabs_reference_profile(
        eqft.DoRAConfig(norm_gradient="full")
    )
    factored_dora = eqft.dora_liu2024_profile(eqft.DoRAConfig(norm_impl="factored"))
    scaling_factored = eqft.dora_factored_2026_profile()
    pissa = eqft.pissa_meng2024_profile()
    vera = eqft.vera_kopiczko2024_profile()
    unshared_vera = eqft.vera_kopiczko2024_profile(eqft.VeRAConfig(shared=False))
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
    fourierft = eqft.fourierft_gao2024_profile(
        eqft.FourierFTConfig(num_coefficients=2, frequency_selection="random", seed=7)
    )
    oft = eqft.oft_qiu2023_profile()
    non_oft = eqft.oft_qiu2023_profile(
        eqft.OrthogonalAdapterConfig(parameterization="butterfly_cayley", block_size=2)
    )
    boft = eqft.boft_liu2024_profile(
        eqft.OrthogonalAdapterConfig(parameterization="butterfly_cayley", block_size=2)
    )
    randlora = eqft.randlora_profile(eqft.RandLoRAConfig(seed=None))

    assert lora_default.fidelity == "safe_default"
    assert "Q/V-only" in lora_default.known_deviations[0]
    assert lora_qv.fidelity == "reference_implementation"
    assert lora_qv.target_spec["tags_any"] == ("attention.q", "attention.v")
    assert lora_qv.target_spec["target_kind"] == "projection_segment"
    assert rslora.fidelity == "reference_implementation"
    assert rslora.config["scaling"] == "alpha_over_sqrt_r"
    assert ordinary_scaled_rslora.fidelity == "experimental"
    assert "alpha_over_sqrt_r" in ordinary_scaled_rslora.known_deviations[0]
    assert dora_paper.fidelity == "paper_exact"
    assert dora_paper.config["norm_gradient"] == "full"
    assert dora_reference.fidelity == "reference_implementation"
    assert dora_reference.config["norm_gradient"] == "detached"
    assert dora_reference.config["norm_impl"] == "dense"
    assert full_gradient_reference.fidelity == "experimental"
    assert "stop-gradient" in full_gradient_reference.known_deviations[0]
    assert factored_dora.fidelity == "paper_exact"
    assert factored_dora.config["norm_impl"] == "factored"
    assert scaling_factored.fidelity == "experimental"
    assert scaling_factored.config["norm_impl"] == "factored"
    assert scaling_factored.config["norm_gradient"] == "detached"
    assert "Triton fused runtime" in scaling_factored.known_deviations[-1]
    assert pissa.fidelity == "reference_implementation"
    assert pissa.required_artifacts == ("principal_svd_or_product",)
    assert vera.fidelity == "safe_default"
    assert "shape-compatible" in vera.known_deviations[0]
    assert unshared_vera.fidelity == "experimental"
    assert "shared=False" in unshared_vera.known_deviations[1]
    assert lora_fa.fidelity == "reference_implementation"
    assert historical.fidelity == "experimental"
    assert historical.known_deviations
    assert eva.reference_ids == ("paischer2024_eva", "ml_jku_eva")
    assert eva.required_artifacts == ("activation_svd",)
    assert fourierft.fidelity == "reference_implementation"
    assert fourierft.reference_ids == (
        "gao2024_fourierft",
        "chaos96_fourierft",
        "hf_peft_fourierft",
    )
    assert oft.fidelity == "reference_implementation"
    assert oft.reference_ids == ("qiu2023_oft", "zqiu24_oft", "hf_peft_oft")
    assert non_oft.fidelity == "experimental"
    assert "parameterization='cayley'" in non_oft.known_deviations[0]
    assert boft.fidelity == "reference_implementation"
    assert boft.reference_ids == ("liu2024_boft", "wy1iu_butterfly_oft", "hf_peft_boft")
    assert randlora.fidelity == "experimental"
    assert randlora.reference_ids == ("albert2025_randlora", "paulalbert31_randlora")
    assert "explicit seed" in randlora.known_deviations[0]
    assert lora_fa.target_spec["tags_any"] == ("attention.qkv", "attention.proj")
    assert houlsby.fidelity == "safe_default"
    assert houlsby.config["placement"] == "both"
    assert "truncated-normal" in houlsby.known_deviations[0]
    assert single_site_adapter.fidelity == "experimental"
    assert "both attention and feed-forward" in single_site_adapter.known_deviations[1]
    assert adapterfusion.fidelity == "reference_implementation"
    assert adapterfusion.required_artifacts == ("trained_adapter_bank",)
    assert adapterfusion.reference_ids == (
        "pfeiffer2021_adapterfusion",
        "adapterhub_adapters",
    )
    assert unfrozen_fusion.fidelity == "experimental"
    assert "second stage" in unfrozen_fusion.known_deviations[0]
    assert adaptformer.fidelity == "paper_exact"
    assert adaptformer.reference_ids == (
        "chen2022_adaptformer",
        "shoufachen_adaptformer",
    )
    assert adaptformer.config["activation"] == "relu"
    assert adaptformer.config["scale_init"] == 0.1
    assert adaptformer.config["scale_trainable"] is False
    assert equimo_adaptformer.fidelity == "experimental"
    assert "ReLU" in equimo_adaptformer.known_deviations[0]
    assert vpt_deep.fidelity == "reference_implementation"
    assert vpt_deep.config["depth"] == "deep"
    assert vpt_deep.config["prepend_to"] == "after_cls"
    assert vpt_wrong_insert.fidelity == "experimental"
    assert "after the class token" in vpt_wrong_insert.known_deviations[0]
    assert vpt_shallow.fidelity == "reference_implementation"
    assert vpt_shallow.config["depth"] == "shallow"
    assert soft_prompt.fidelity == "safe_default"
    assert soft_prompt.config["prepend_to"] == "input"
    assert deep_soft_prompt.fidelity == "experimental"
    assert "model input" in deep_soft_prompt.known_deviations[0]
    assert ptuning_v2.fidelity == "safe_default"
    assert ptuning_v2.config["depth"] == "all"
    assert ptuning_v2.config["reparameterizer"] == "none"
    assert ptuning_shared.fidelity == "experimental"
    assert "layer-specific" in ptuning_shared.known_deviations[0]
    assert prefix_tuning.fidelity == "reference_implementation"
    assert prefix_tuning.config["target"] == ("attention.k", "attention.v")
    assert prefix_no_projection.fidelity == "experimental"
    assert "projection" in prefix_no_projection.known_deviations[0]
    assert ia3.fidelity == "reference_implementation"
    assert ia3.config["target"]["tags_any"] == (
        "attention.k",
        "attention.v",
        "mlp.hidden",
    )
    assert ia3_missing_value.fidelity == "experimental"
    assert "key/value" in ia3_missing_value.known_deviations[0]
    assert ssf.fidelity == "reference_implementation"
    assert ssf.config["init"] == "normal"
    assert ssf.config["init_std"] == 0.02
    assert "attention.qkv" in ssf.target_spec["tags_any"]
    assert broad_ssf.fidelity == "experimental"
    assert "broad tags" in broad_ssf.known_deviations[0]


def test_profiles_do_not_claim_exactness_for_unpinned_random_choices():
    profile = eqft.fourierft_gao2024_profile(
        eqft.FourierFTConfig(num_coefficients=2, frequency_selection="random")
    )
    boft = eqft.boft_liu2024_profile(
        eqft.OrthogonalAdapterConfig(parameterization="cayley")
    )

    assert profile.fidelity == "experimental"
    assert "explicit seed" in profile.known_deviations[0]
    assert boft.fidelity == "experimental"
    assert boft.known_deviations
