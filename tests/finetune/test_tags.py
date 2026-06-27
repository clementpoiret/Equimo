"""Fine-tuning semantic-tag tests."""

from __future__ import annotations

import equinox as eqx
import equimo.finetune as eqft
import jax
import jax.numpy as jnp
import jax.random as jr
from equimo.finetune.audio import selectors as audio_selectors
from equimo.finetune.language import selectors as language_selectors
from equimo.finetune.vision import selectors as vision_selectors
from equimo.vision.models.vit import dinov2_vits14, dinov2_vits14_reg

from fixtures import TinyVisionTransformer


def _infos_by_path(model):
    return {eqft.path_to_str(info.path): info for info in eqft.iter_param_infos(model)}


class _SpecialTokenLeaves(eqx.Module):
    mask_token: jax.Array
    dist_token: jax.Array
    token_embed: jax.Array


def test_canonical_tags_include_selector_defaults():
    required = {
        "embedding",
        "embedding.patch",
        "embedding.position",
        "embedding.class_token",
        "embedding.register_token",
        "embedding.mask_token",
        "embedding.token",
        "block",
        "block.attention",
        "block.attention.qkv",
        "attention",
        "attention.qkv",
        "attention.proj",
        "block.mlp",
        "mlp",
        "mlp.hidden",
        "mlp.fc1",
        "mlp.fc2",
        "norm",
        "head",
        "audio.patch_embed",
        "text.embedding",
        "tabular.encoder",
    }

    assert required <= set(eqft.CANONICAL_TAGS)


def test_semantic_tags_for_tiny_vit(tiny_vision_transformer):
    infos = _infos_by_path(tiny_vision_transformer)

    assert "embedding.patch" in infos["patch_embed.proj.weight"].tags
    assert "embedding.position" in infos["pos_embed"].tags
    assert "embedding.class_token" in infos["cls_token"].tags
    assert {
        "block",
        "block.0",
        "attention",
        "block.attention",
        "attention.qkv",
    } <= infos["blocks.0.attn.qkv.weight"].tags
    assert infos["blocks.0.attn.qkv.weight"].depth == 0
    assert "attention.proj" in infos["blocks.1.attn.proj.bias"].tags
    assert {"mlp", "block.mlp", "mlp.hidden", "mlp.fc1"} <= infos[
        "blocks.1.mlp.fc1.weight"
    ].tags
    assert "mlp.fc2" in infos["blocks.1.mlp.fc2.bias"].tags
    assert {"norm", "block.norm.post", "bias"} <= infos["blocks.1.norm2.bias"].tags
    assert "head" in infos["head.weight"].tags


def test_nested_vit_paths_use_inner_block_depth():
    path = ("blocks", 0, "blocks", 11, "attn", "qkv", "weight")
    tags = eqft.canonical_tags_for_path(path)

    assert eqft.infer_depth(path) == 11
    assert "block.11" in tags
    assert "block.0" not in tags


def test_canonical_tags_cover_prefixed_position_and_plural_token_names():
    assert "embedding.position" in eqft.canonical_tags_for_path(
        ("global_pos_embed", "weight")
    )
    assert "embedding.position" in eqft.canonical_tags_for_path(
        ("local_pos_embed", "patch_rope", "freqs")
    )
    assert "embedding.class_token" in eqft.canonical_tags_for_path(("cls_tokens",))
    assert "embedding.register_token" in eqft.canonical_tags_for_path(
        ("register_tokens",)
    )


def test_dinov2_global_pos_embed_is_no_decay_without_register_tokens():
    model = dinov2_vits14(pretrained=False, key=jr.PRNGKey(0))

    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
        labels=eqft.LLRDConfig(),
    )

    info = plan.param_info.global_pos_embed.weight
    assert "embedding.position" in info.tags
    assert info.label == "embed_no_decay"
    assert info.weight_decay is False


def test_dinov2_global_pos_embed_and_register_tokens_are_no_decay():
    model = dinov2_vits14_reg(pretrained=False, key=jr.PRNGKey(0))

    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
        labels=eqft.LLRDConfig(),
    )

    pos_info = plan.param_info.global_pos_embed.weight
    reg_info = plan.param_info.reg_tokens
    assert "embedding.position" in pos_info.tags
    assert pos_info.label == "embed_no_decay"
    assert pos_info.weight_decay is False
    assert "embedding.register_token" in reg_info.tags
    assert reg_info.label == "embed_no_decay"
    assert reg_info.weight_decay is False


def test_register_token_tags_and_labels():
    model = TinyVisionTransformer(num_reg_tokens=2)
    infos = _infos_by_path(model)

    assert "embedding.register_token" in infos["reg_tokens"].tags
    paths = eqft.resolve_target_paths(
        model,
        eqft.TargetSpec(tags_any=("embedding.register_token",)),
    )
    assert paths == ("reg_tokens",)

    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="full"),
        labels=eqft.LLRDConfig(),
    )
    assert plan.param_info.reg_tokens.label == "embed_no_decay"
    assert plan.param_info.reg_tokens.weight_decay is False


def test_make_tag_tree_records_roles(tiny_vision_transformer):
    tag_tree = eqft.make_tag_tree(tiny_vision_transformer)

    assert tag_tree.blocks[0].attn.qkv.weight.role == "attention.qkv"
    assert tag_tree.head.bias.role == "head"


def test_special_token_roles_are_inferred():
    model = _SpecialTokenLeaves(
        mask_token=jnp.zeros((1, 4), dtype=jnp.float32),
        dist_token=jnp.zeros((1, 4), dtype=jnp.float32),
        token_embed=jnp.zeros((8, 4), dtype=jnp.float32),
    )
    infos = _infos_by_path(model)

    assert infos["mask_token"].role == "embedding.mask_token"
    assert infos["dist_token"].role == "embedding.distillation_token"
    assert infos["token_embed"].role == "embedding.token"


def test_model_family_tag_adapters(tiny_ast_like_encoder, tiny_text_encoder):
    assert "embedding.patch" in vision_selectors.vision_tags_for_path(
        ("patch_embed", "proj", "weight")
    )

    audio_paths = audio_selectors.resolve_target_paths(
        tiny_ast_like_encoder,
        eqft.TargetSpec(tags_any=("audio.patch_embed",)),
    )
    assert audio_paths == ("patch_embed.proj.weight", "patch_embed.proj.bias")

    text_paths = language_selectors.resolve_target_paths(
        tiny_text_encoder,
        eqft.TargetSpec(tags_any=("text.embedding",)),
    )
    assert text_paths == ("token_embed.weight",)


def test_conv_stage_tags_for_convnext_like(tiny_convnext_like):
    infos = _infos_by_path(tiny_convnext_like)

    assert "stem" in infos["stem.weight"].tags
    assert {"stage", "stage.block", "conv.depthwise"} <= infos[
        "stages.0.blocks.0.dwconv.weight"
    ].tags
    assert {"conv", "conv.pointwise"} <= infos["stages.0.blocks.0.pwconv1.weight"].tags
