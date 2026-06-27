"""Semantic tags for fine-tuning selector infrastructure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu

from ._typing import Path, PyTree
from .config import ParamInfo
from .paths import iter_param_leaves, key_path_to_path

Tagger = Callable[[Path, Any], Iterable[str]]

CANONICAL_TAGS: tuple[str, ...] = (
    "embedding",
    "embedding.patch",
    "embedding.position",
    "embedding.class_token",
    "embedding.register_token",
    "embedding.distillation_token",
    "embedding.mask_token",
    "embedding.token",
    "block",
    "block.attention",
    "block.attention.qkv",
    "block.attention.q",
    "block.attention.k",
    "block.attention.v",
    "block.attention.proj",
    "attention",
    "attention.qkv",
    "attention.q",
    "attention.k",
    "attention.v",
    "attention.proj",
    "block.mlp",
    "block.mlp.hidden",
    "block.mlp.fc1",
    "block.mlp.fc2",
    "block.norm.pre",
    "block.norm.post",
    "block.drop_path",
    "block.layer_scale",
    "mlp",
    "mlp.hidden",
    "mlp.fc1",
    "mlp.fc2",
    "stem",
    "stage",
    "stage.block",
    "stage.downsample",
    "conv",
    "conv.depthwise",
    "conv.pointwise",
    "norm",
    "pool",
    "head",
    "head.classifier",
    "head.projection",
    "audio.frontend",
    "audio.spectrogram",
    "audio.patch_embed",
    "audio.frame_encoder",
    "audio.transformer.block",
    "audio.pooling",
    "audio.ctc_head",
    "audio.tagging_head",
    "text.embedding",
    "text.position",
    "text.token_type",
    "text.block",
    "text.attention.qkv",
    "text.attention.q",
    "text.attention.k",
    "text.attention.v",
    "text.attention.proj",
    "text.mlp.fc1",
    "text.mlp.fc2",
    "text.pooler",
    "text.projection_head",
    "tabular.encoder",
    "tabular.tokenizer",
    "tabular.block",
    "tabular.head",
)


def canonical_tags_for_path(path: Path, leaf: Any | None = None) -> frozenset[str]:
    """Infer canonical semantic tags from an Equimo-style parameter path."""

    del leaf
    parts = tuple(str(part) for part in path)
    tags: set[str] = set()

    if "patch_embed" in parts:
        tags.update(("embedding", "embedding.patch"))
    if any(
        part in {"pos_embed", "position_embed", "position_embedding"} for part in parts
    ):
        tags.update(("embedding", "embedding.position"))
    if "cls_token" in parts:
        tags.update(("embedding", "embedding.class_token"))
    if any(
        part in {"reg_token", "reg_tokens", "register_token", "register_tokens"}
        for part in parts
    ):
        tags.update(("embedding", "embedding.register_token"))
    if "dist_token" in parts:
        tags.update(("embedding", "embedding.distillation_token"))
    if "mask_token" in parts:
        tags.update(("embedding", "embedding.mask_token"))
    if "token_embed" in parts or "token_embedding" in parts:
        tags.update(("embedding", "embedding.token"))

    depth = infer_depth(path)
    if depth is not None:
        tags.add("block")
        tags.add(f"block.{depth}")

    _add_attention_tags(parts, tags)
    _add_mlp_tags(parts, tags)
    _add_norm_tags(parts, tags)
    _add_conv_stage_tags(parts, tags)
    _add_peft_tags(parts, tags)

    if "head" in parts or "classifier" in parts:
        tags.update(("head", "head.classifier"))
    if "projection_head" in parts:
        tags.update(("head", "head.projection"))
    if "pool" in parts or "pooler" in parts:
        tags.add("pool")
    if "encoder" in parts:
        tags.add("tabular.encoder")
    if "tokenizer" in parts:
        tags.add("tabular.tokenizer")

    if parts[-1:] == ("bias",):
        tags.add("bias")
    elif parts[-1:] in (("weight",), ("scale",)):
        tags.add("weight")

    return frozenset(tags)


def infer_depth(path: Path) -> int | None:
    """Infer semantic block depth from common Equimo path shapes."""

    parts = tuple(str(part) for part in path)
    depth = infer_block_depth(path)
    if depth is not None:
        return depth

    for index, part in enumerate(parts[:-1]):
        if part == "stages":
            stage = _maybe_int(parts[index + 1])
            if stage is not None:
                return stage

    return None


def infer_block_depth(path: Path) -> int | None:
    """Infer the innermost block index from a parameter path."""

    parts = tuple(str(part) for part in path)
    return _last_indexed_depth(parts, {"blocks", "block"})


def make_tag_tree(
    tree: PyTree,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Replace parameter-like leaves with tagged ``ParamInfo`` records."""

    filtered = eqx.filter(tree, eqx.is_inexact_array)

    def make_info(key_path: tuple[Any, ...], leaf: Any) -> ParamInfo:
        return make_param_info(key_path_to_path(key_path), leaf, tagger=tagger)

    return jtu.tree_map_with_path(make_info, filtered)


def iter_param_infos(
    tree: PyTree,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> tuple[ParamInfo, ...]:
    """Return tagged ``ParamInfo`` records for inexact array leaves."""

    return tuple(
        make_param_info(path, leaf, tagger=tagger)
        for path, leaf in iter_param_leaves(tree)
    )


def make_param_info(
    path: Path,
    leaf: Any,
    *,
    tagger: Tagger = canonical_tags_for_path,
) -> ParamInfo:
    """Build one tagged ``ParamInfo`` record."""

    tags = frozenset(tagger(path, leaf))
    path_string = ".".join(str(part) for part in path)
    return ParamInfo(
        path=path,
        logical_id=path_string,
        tags=tags,
        role=infer_role(tags),
        depth=infer_depth(path),
        is_array=eqx.is_array(leaf),
        is_inexact_array=eqx.is_inexact_array(leaf),
    )


def infer_role(tags: Iterable[str]) -> str:
    """Return a coarse role name from semantic tags."""

    tag_set = frozenset(tags)
    for tag, role in (
        ("lora.factor_A", "lora.factor_A"),
        ("lora.factor_B", "lora.factor_B"),
        ("lora_fa.factor_B", "lora_fa.factor_B"),
        ("adalora.singular", "adalora.singular"),
        ("fourierft", "fourierft"),
        ("orthogonal", "orthogonal"),
        ("randlora.scale", "randlora.scale"),
        ("randlora", "randlora"),
        ("lora", "lora"),
        ("head", "head"),
        ("attention.qkv", "attention.qkv"),
        ("attention.proj", "attention.proj"),
        ("mlp.hidden", "mlp.hidden"),
        ("mlp.fc1", "mlp.fc1"),
        ("mlp.fc2", "mlp.fc2"),
        ("norm", "norm"),
        ("embedding.patch", "embedding.patch"),
        ("embedding.position", "embedding.position"),
        ("embedding.class_token", "embedding.class_token"),
        ("embedding.register_token", "embedding.register_token"),
        ("block", "block"),
    ):
        if tag in tag_set:
            return role
    return ""


def merge_taggers(*taggers: Tagger) -> Tagger:
    """Combine multiple taggers into one tagger."""

    def merged(path: Path, leaf: Any) -> frozenset[str]:
        tags: set[str] = set()
        for tagger in taggers:
            tags.update(tagger(path, leaf))
        return frozenset(tags)

    return merged


def _add_attention_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    for index, part in enumerate(parts[:-1]):
        if part not in {"attn", "attention", "self_attn", "self_attention"}:
            continue
        tags.update(("attention", "block.attention"))
        next_part = parts[index + 1]
        if next_part == "qkv":
            tags.update(("attention.qkv", "block.attention.qkv"))
        elif next_part in {"proj", "out_proj", "projection"}:
            tags.update(("attention.proj", "block.attention.proj"))
        elif next_part in {"q", "query"}:
            tags.update(("attention.q", "block.attention.q"))
        elif next_part in {"k", "key"}:
            tags.update(("attention.k", "block.attention.k"))
        elif next_part in {"v", "value"}:
            tags.update(("attention.v", "block.attention.v"))


def _add_mlp_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    mlp_like = bool({"mlp", "ffn", "feed_forward"}.intersection(parts))
    if not mlp_like:
        return
    tags.update(("mlp", "block.mlp"))
    if "fc1" in parts:
        tags.update(("mlp.hidden", "block.mlp.hidden", "mlp.fc1", "block.mlp.fc1"))
    if "fc2" in parts:
        tags.update(("mlp.fc2", "block.mlp.fc2"))


def _add_norm_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    norm_parts = {"norm", "norm1", "norm2", "ln", "layer_norm"}
    matched = norm_parts.intersection(parts)
    if not matched:
        return

    tags.add("norm")
    if "norm1" in matched:
        tags.add("block.norm.pre")
    if "norm2" in matched:
        tags.add("block.norm.post")


def _add_conv_stage_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    if "stem" in parts:
        tags.add("stem")
    if "stages" in parts or "stage" in parts:
        tags.add("stage")
        if "blocks" in parts or "block" in parts:
            tags.add("stage.block")
        if "downsample" in parts:
            tags.add("stage.downsample")
    conv_parts = {part for part in parts if "conv" in part}
    if conv_parts:
        tags.add("conv")
    if any(part in {"dwconv", "depthwise_conv", "depthwise"} for part in parts):
        tags.add("conv.depthwise")
    if any(
        part.startswith("pwconv") or part in {"pointwise_conv", "pointwise"}
        for part in parts
    ):
        tags.add("conv.pointwise")


def _add_peft_tags(parts: tuple[str, ...], tags: set[str]) -> None:
    if "frozen_A" in parts:
        tags.update(("lora_fa", "peft.metadata"))
    if "lora_fa_B" in parts:
        tags.update(("peft", "lora_fa", "lora_fa.factor_B"))
    if "lora_A" in parts:
        tags.update(("lora", "lora.factor_A"))
    if "lora_B" in parts:
        tags.update(("lora", "lora.factor_B"))
    if "coefficients_real" in parts or "coefficients_imag" in parts:
        tags.update(("peft", "fourierft"))
    if "frequency_indices" in parts:
        tags.add("peft.metadata")
    if "skew" in parts:
        tags.update(("peft", "orthogonal"))
    if "basis_scales" in parts:
        tags.update(("peft", "randlora", "randlora.scale"))
    if "random_A" in parts or "random_B" in parts:
        tags.update(("randlora", "peft.metadata"))
    if "singular" in parts and "adalora" in parts:
        tags.update(("adalora", "adalora.singular"))
    if "rank_mask" in parts:
        tags.update(("lora", "lora.rank_mask"))
    if "base_weight_delta" in parts:
        tags.add("peft.metadata")
    if "magnitude" in parts:
        tags.update(("dora", "dora.magnitude"))
    if "adapter" in parts or "adapters" in parts:
        tags.add("adapter")
    if "adapter_fusion" in parts:
        tags.update(("adapter", "adapter_fusion"))
    if "prompt" in parts or "prompts" in parts:
        tags.add("prompt")
    if (
        "prefix" in parts
        or "prefixes" in parts
        or any(part.startswith("prefix_") for part in parts)
    ):
        tags.add("prefix")
    if "ia3" in parts:
        tags.add("ia3")
    if "vera_input_scale" in parts or "vera_output_scale" in parts:
        tags.add("vera")
    if "scale_shift" in parts:
        tags.add("scale_shift")


def _maybe_int(value: str) -> int | None:
    return int(value) if value.isdecimal() else None


def _last_indexed_depth(parts: tuple[str, ...], names: set[str]) -> int | None:
    depth = None
    for index, part in enumerate(parts[:-1]):
        if part in names:
            value = _maybe_int(parts[index + 1])
            if value is not None:
                depth = value
    return depth


__all__ = (
    "CANONICAL_TAGS",
    "Tagger",
    "canonical_tags_for_path",
    "infer_depth",
    "infer_role",
    "iter_param_infos",
    "make_param_info",
    "make_tag_tree",
    "merge_taggers",
)
