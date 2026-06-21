"""Fine-tuning scaffolding tests."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

import equimo.finetune as eqft

from fixtures import (
    EXPECTED_PARAM_COUNTS,
    TinyASTLikeEncoder,
    TinyConvNeXtLike,
    TinyLinearMLP,
    TinyTextEncoder,
    TinyVisionTransformer,
    count_params,
    extract_paths,
)


def test_import_equimo_finetune():
    assert eqft.TargetSpec(tags_any=("attention.qkv",)).tags_any == ("attention.qkv",)
    assert eqft.TrainableSpec(mode="head").mode == "head"
    assert eqft.LLRDConfig().decay == 0.75


def test_required_public_api_exports():
    required = {
        "AdapterBankConfig",
        "AdapterFusion",
        "AdapterFusionConfig",
        "AdapterRecipe",
        "BitFitConfig",
        "ContinuedSSLAdaptationConfig",
        "ContinuedSSLPlan",
        "PTuningV2Config",
        "DenseFeatureAdapter",
        "DoRARecipe",
        "EWCConfig",
        "FineTuneBundleError",
        "FineTuneRecipe",
        "GlobalAveragePool",
        "GreedySoupConfig",
        "HeadOnlyModel",
        "HeadPlusNormConfig",
        "HeadSpec",
        "LinearProbeConfig",
        "LinearProbeRecipe",
        "LoRAPlusLabelConfig",
        "LoRARecipe",
        "MixoutConfig",
        "PEFTConfig",
        "OutputAdapterModule",
        "PartialUnfreezeConfig",
        "ParallelAdapterConfig",
        "PrefixProjection",
        "SAMMetadata",
        "SoftPromptConfig",
        "SupervisedAfterSSLConfig",
        "VPTDeepRecipe",
        "VPTShallowRecipe",
        "adapter_fusion_trainable_spec",
        "adapter_norm_loss",
        "apply_adapter_fusion",
        "audio",
        "bitfit_trainable_spec",
        "configure_adapter_bank",
        "continued_ssl_adaptation",
        "delta_attention_aux_loss_spec",
        "delta_attention_loss",
        "ewc_loss",
        "iter_ia3_modules",
        "iter_scale_shift_wrappers",
        "language",
        "lora_transformer",
        "lora_transformer_all_linear",
        "merge_dora",
        "merge_ia3",
        "merge_scale_shift",
        "merge_vera",
        "add_adapter",
        "set_active_adapter",
        "save_finetune_bundle",
        "load_finetune_bundle",
        "merge_and_save",
        "mixout_leaf",
        "mixout_tree",
        "partial_unfreeze",
        "recipes",
        "tabular",
        "task_adapter_bank",
        "task_vector_norm_loss",
        "vision",
        "vpt_deep",
    }

    assert not {name for name in required if not hasattr(eqft, name)}


def test_recipe_namespaces_export_direct_helpers():
    assert eqft.recipes.lora_transformer is eqft.lora_transformer
    assert hasattr(eqft.recipes, "lora_transformer_all_linear")
    assert hasattr(eqft.recipes, "vpt_deep")
    assert hasattr(eqft.recipes, "task_adapter_bank")
    assert hasattr(eqft.vision, "full_ft_vit_llrd")
    assert hasattr(eqft.vision, "dense_feature_adapter")
    assert hasattr(eqft.vision, "prompts")
    assert hasattr(eqft.audio, "adapter_ast")
    assert hasattr(eqft.language, "prefix_encoder")
    assert hasattr(eqft.tabular, "head_only")


def test_spec_config_fields_are_public():
    recipe = eqft.FineTuneRecipe(
        name="linear_probe",
        method="head",
        head=eqft.HeadSpec(kind="linear"),
        peft=None,
        trainable=eqft.TrainableSpec(mode="head"),
        labels=None,
    )

    assert recipe.external_hints == {}
    assert eqft.LoRAConfig(fan_in_fan_out=True).fan_in_fan_out
    assert eqft.DoRAConfig(dropout=0.1, train_base=True, mergeable=False).dropout == 0.1
    assert eqft.AdapterConfig(residual_scale_init=0.5).train_base is False
    assert eqft.AdaptFormerConfig(train_head=False).train_head is False
    assert eqft.ParallelAdapterConfig(branch="mlp").placement == "parallel"
    assert eqft.AdapterBankConfig(active="task").missing_adapter_policy == "error"
    assert eqft.AdapterFusionConfig(fusion_dropout=0.1).freeze_task_adapters
    assert eqft.adapter_fusion_trainable_spec().target.tags_any == ("adapter_fusion",)
    assert eqft.BitFitConfig(include_norm_bias=False).train_bias
    assert eqft.MixoutConfig(anchor="pretrained").p == 0.1
    assert eqft.LinearProbeConfig().cache_features is False
    assert eqft.HeadPlusNormConfig().bn_stats_policy == "frozen"
    assert eqft.PartialUnfreezeConfig().fraction == 1 / 3
    assert eqft.LoRAPlusLabelConfig().label_A == "lora_A"
    assert eqft.LoRARecipe.hard_task().rank == 16
    assert eqft.LoRARecipe.tiny_data().rank == 4
    assert eqft.StaticRankMaskedLoRAConfig().rank_mask_init == "all_active"
    assert eqft.DoRARecipe().external_lr_hint == "slightly_lower_than_lora"
    assert eqft.AdapterRecipe.strong().placement == "both"
    assert eqft.LinearProbeRecipe().feature_norm == "l2_or_standardize"
    assert eqft.VPTShallowRecipe().num_tokens == 50
    assert eqft.VPTDeepRecipe().depth == "deep"
    assert eqft.WiSEFTConfig().mask == "shared_backbone"
    assert eqft.UniformSoupConfig().strict_shapes
    assert eqft.GreedySoupConfig().start == "best_model"
    assert eqft.ContinuedSSLAdaptationConfig().save_stage == "continued_ssl_delta"
    assert eqft.SupervisedAfterSSLConfig().reuse_ssl_delta
    assert eqft.TIESConfig().merge == "disjoint_mean"
    assert eqft.TaskVectorConfig().mask == "floating_backbone"
    assert eqft.DARETransform().scope == "per_tensor"
    assert eqft.BreadcrumbsConfig().rescale is False
    assert eqft.FisherMergeConfig().normalize_fisher
    assert eqft.RegMeanConfig().require_input_covariances
    assert eqft.SAMMetadata().external_only
    assert eqft.LLRDConfig.vit_base().decay == 0.65
    assert eqft.LLRDConfig.vit_large_or_huge().decay == 0.75
    assert eqft.LLRDConfig.audio_transformer().decay == 0.75
    assert eqft.PromptConfig(init="normal", train_head=False).train_head is False
    assert eqft.SoftPromptConfig(num_tokens=2).prepend_to == "input"
    assert eqft.PTuningV2Config(share_across_layers=True).depth == "all"
    assert eqft.PrefixConfig(prefix_dropout=0.1, train_head=False).prefix_dropout == 0.1
    assert eqft.IA3Config(axis="feature", mergeable=True).mergeable
    assert eqft.ScaleShiftConfig(axis="channel", mergeable=True).axis == "channel"
    assert eqft.ScaleShiftConfig.convnet().target.tags_any == (
        "conv",
        "stage.block",
        "norm",
    )
    vera = eqft.VeRAConfig(
        seed_required=True,
        frozen_A_init="kaiming_uniform",
        trainable_input_scale_init=0.5,
    )
    assert vera.target.tags_any == ("attention.qkv", "attention.proj")
    assert vera.trainable_input_scale_init == 0.5


def test_tiny_fixture_param_counts(finetune_key):
    models = {
        "tiny_ast_like_encoder": TinyASTLikeEncoder(key=finetune_key),
        "tiny_convnext_like": TinyConvNeXtLike(key=finetune_key),
        "tiny_linear_mlp": TinyLinearMLP(key=finetune_key),
        "tiny_text_encoder": TinyTextEncoder(key=finetune_key),
        "tiny_vision_transformer": TinyVisionTransformer(key=finetune_key),
    }

    assert {
        name: count_params(model) for name, model in models.items()
    } == EXPECTED_PARAM_COUNTS


def test_tiny_fixture_paths_are_predictable(tiny_vision_transformer, tiny_text_encoder):
    vit_paths = extract_paths(tiny_vision_transformer)
    text_paths = extract_paths(tiny_text_encoder)

    assert "patch_embed.proj.weight" in vit_paths
    assert "blocks.0.attn.qkv.weight" in vit_paths
    assert "blocks.1.mlp.fc2.bias" in vit_paths
    assert "head.bias" in vit_paths
    assert "token_embed.weight" in text_paths
    assert "blocks.0.attn.proj.weight" in text_paths


class _TiedLeaves(eqx.Module):
    left: jnp.ndarray
    right: jnp.ndarray


def test_param_identities_mark_tied_alias_groups():
    shared = jnp.ones((2,), dtype=jnp.float32)
    model = _TiedLeaves(shared, shared)

    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
    )

    assert plan.identities.left.alias_group is not None
    assert plan.identities.left.alias_group == plan.identities.right.alias_group


def test_finetune_core_does_not_import_optimizer_libraries():
    finetune_root = Path(eqft.__file__).parent
    source = "\n".join(path.read_text() for path in finetune_root.rglob("*.py"))

    assert "import optax" not in source
    assert "import rollfast" not in source
