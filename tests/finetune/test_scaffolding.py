"""Fine-tuning scaffolding tests."""

from __future__ import annotations

from pathlib import Path

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
    assert eqft.TargetSpec(tags=("attention.qkv",)).tags == ("attention.qkv",)
    assert eqft.TrainableSpec(mode="head").mode == "head"
    assert eqft.LLRDConfig().decay == 0.75


def test_required_public_api_exports():
    required = {
        "HeadSpec",
        "merge_dora",
        "add_adapter",
        "set_active_adapter",
        "save_finetune_bundle",
        "load_finetune_bundle",
        "merge_and_save",
    }

    assert not {name for name in required if not hasattr(eqft, name)}


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


def test_finetune_core_does_not_import_optimizer_libraries():
    finetune_root = Path(eqft.__file__).parent
    source = "\n".join(path.read_text() for path in finetune_root.rglob("*.py"))

    assert "import optax" not in source
    assert "import rollfast" not in source
