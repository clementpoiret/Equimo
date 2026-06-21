"""Fine-tuning selector-resolution tests."""

from __future__ import annotations

import pytest

import equimo.finetune as eqft


def test_selector_tags_qkv(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags=("attention.qkv",)),
    )

    assert paths == (
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.qkv.bias",
        "blocks.1.attn.qkv.weight",
        "blocks.1.attn.qkv.bias",
    )


def test_selector_broad_attention_and_mlp_tags(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags=("attention", "mlp")),
    )

    assert "blocks.0.attn.qkv.weight" in paths
    assert "blocks.0.attn.proj.bias" in paths
    assert "blocks.0.mlp.fc1.weight" in paths
    assert "blocks.1.mlp.fc2.bias" in paths


def test_selector_tags_patch_embed(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags=("embedding.patch",)),
    )

    assert paths == ("patch_embed.proj.weight", "patch_embed.proj.bias")


def test_selector_glob_qkv(tiny_vision_transformer):
    semantic_paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags=("attention.qkv",)),
    )
    glob_paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(include=("*.blocks.*.attn.qkv",)),
    )

    assert glob_paths == semantic_paths


def test_selector_exclude_pos_embed(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(include=("*",), exclude=("*.pos_embed", "*.cls_token")),
    )

    assert "patch_embed.proj.weight" in paths
    assert "head.bias" in paths
    assert "pos_embed" not in paths
    assert "cls_token" not in paths


def test_selector_callable_linear(tiny_linear_mlp):
    paths = eqft.resolve_target_paths(
        tiny_linear_mlp,
        eqft.TargetSpec(predicate=eqft.is_linear),
    )

    assert paths == ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")


def test_selector_empty_raises(tiny_vision_transformer):
    with pytest.raises(ValueError, match="missing.tag"):
        eqft.resolve_target_paths(
            tiny_vision_transformer,
            eqft.TargetSpec(tags=("missing.tag",)),
        )


def test_depth_resolution_for_blocks(tiny_vision_transformer):
    paths = eqft.resolve_target_paths(
        tiny_vision_transformer,
        eqft.TargetSpec(tags=("block",), min_depth=1, max_depth=1),
    )

    assert paths
    assert all(path.startswith("blocks.1.") for path in paths)
    assert not any(path.startswith("blocks.0.") for path in paths)
