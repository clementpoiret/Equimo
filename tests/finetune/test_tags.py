"""Fine-tuning semantic-tag tests."""

from __future__ import annotations

import equimo.finetune as eqft
from equimo.finetune.audio import selectors as audio_selectors
from equimo.finetune.language import selectors as language_selectors
from equimo.finetune.vision import selectors as vision_selectors

from fixtures import TinyVisionTransformer


def _infos_by_path(model):
    return {eqft.path_to_str(info.path): info for info in eqft.iter_param_infos(model)}


def test_canonical_tags_include_selector_defaults():
    assert set(eqft.CANONICAL_TAGS) == {
        "embedding.patch",
        "embedding.position",
        "embedding.class_token",
        "embedding.register_token",
        "block",
        "attention.qkv",
        "attention.proj",
        "mlp.fc1",
        "mlp.fc2",
        "norm",
        "head",
    }


def test_semantic_tags_for_tiny_vit(tiny_vision_transformer):
    infos = _infos_by_path(tiny_vision_transformer)

    assert "embedding.patch" in infos["patch_embed.proj.weight"].tags
    assert "embedding.position" in infos["pos_embed"].tags
    assert "embedding.class_token" in infos["cls_token"].tags
    assert {"block", "block.0", "attention.qkv"} <= infos[
        "blocks.0.attn.qkv.weight"
    ].tags
    assert infos["blocks.0.attn.qkv.weight"].depth == 0
    assert "attention.proj" in infos["blocks.1.attn.proj.bias"].tags
    assert "mlp.fc1" in infos["blocks.1.mlp.fc1.weight"].tags
    assert "mlp.fc2" in infos["blocks.1.mlp.fc2.bias"].tags
    assert {"norm", "block.norm.post", "bias"} <= infos["blocks.1.norm2.bias"].tags
    assert "head" in infos["head.weight"].tags


def test_register_token_tags_and_labels():
    model = TinyVisionTransformer(num_reg_tokens=2)
    infos = _infos_by_path(model)

    assert "embedding.register_token" in infos["reg_tokens"].tags
    paths = eqft.resolve_target_paths(
        model,
        eqft.TargetSpec(tags=("embedding.register_token",)),
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


def test_model_family_tag_adapters(tiny_ast_like_encoder, tiny_text_encoder):
    assert "embedding.patch" in vision_selectors.vision_tags_for_path(
        ("patch_embed", "proj", "weight")
    )

    audio_paths = audio_selectors.resolve_target_paths(
        tiny_ast_like_encoder,
        eqft.TargetSpec(tags=("audio.patch_embed",)),
    )
    assert audio_paths == ("patch_embed.proj.weight", "patch_embed.proj.bias")

    text_paths = language_selectors.resolve_target_paths(
        tiny_text_encoder,
        eqft.TargetSpec(tags=("text.embedding",)),
    )
    assert text_paths == ("token_embed.weight",)
