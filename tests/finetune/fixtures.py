"""Tiny deterministic models and helpers for fine-tuning tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from equimo.finetune import ParamInfo


def _key_or_default(key: jax.Array | None) -> jax.Array:
    return jr.PRNGKey(0) if key is None else key


def _map_tokens(fn: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
    return fn(x) if x.ndim == 1 else jax.vmap(fn)(x)


class TinyPatchEmbed(eqx.Module):
    """Linear patch embedding with a predictable ``patch_embed.proj`` path."""

    proj: eqx.nn.Linear

    def __init__(self, in_features: int, dim: int, *, key: jax.Array):
        self.proj = eqx.nn.Linear(in_features, dim, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj(x)


class TinyAttention(eqx.Module):
    """Fused-QKV attention-shaped module for selector tests."""

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(self, dim: int, *, key: jax.Array):
        qkv_key, proj_key = jr.split(key, 2)
        self.qkv = eqx.nn.Linear(dim, dim * 3, key=qkv_key)
        self.proj = eqx.nn.Linear(dim, dim, key=proj_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        qkv = _map_tokens(self.qkv, x)
        _, _, value = jnp.split(qkv, 3, axis=-1)
        return _map_tokens(self.proj, value)


class TinyMLP(eqx.Module):
    """Transformer MLP-shaped module with ``fc1`` and ``fc2`` leaves."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden_dim: int, *, key: jax.Array):
        fc1_key, fc2_key = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(dim, hidden_dim, key=fc1_key)
        self.fc2 = eqx.nn.Linear(hidden_dim, dim, key=fc2_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = _map_tokens(self.fc1, x)
        x = jax.nn.gelu(x)
        return _map_tokens(self.fc2, x)


class TinyTransformerBlock(eqx.Module):
    """Transformer block with stable ``norm``, ``attn``, and ``mlp`` paths."""

    norm1: eqx.nn.LayerNorm
    attn: TinyAttention
    norm2: eqx.nn.LayerNorm
    mlp: TinyMLP

    def __init__(self, dim: int, hidden_dim: int, *, key: jax.Array):
        attn_key, mlp_key = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, key=attn_key)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.mlp = TinyMLP(dim, hidden_dim, key=mlp_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(_map_tokens(self.norm1, x))
        return x + self.mlp(_map_tokens(self.norm2, x))


class TinyVisionTransformer(eqx.Module):
    """ViT-shaped fixture with patch, position, class-token, block, and head paths."""

    dim: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)

    patch_embed: TinyPatchEmbed
    cls_token: jax.Array
    pos_embed: jax.Array
    blocks: tuple[TinyTransformerBlock, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(
        self,
        *,
        dim: int = 4,
        hidden_dim: int = 8,
        depth: int = 2,
        patch_features: int = 3,
        num_patches: int = 2,
        num_classes: int = 2,
        key: jax.Array | None = None,
    ):
        key = _key_or_default(key)
        patch_key, block_key, head_key = jr.split(key, 3)
        self.dim = dim
        self.num_patches = num_patches
        self.patch_embed = TinyPatchEmbed(patch_features, dim, key=patch_key)
        self.cls_token = jnp.zeros((1, dim), dtype=jnp.float32)
        self.pos_embed = jnp.zeros((num_patches + 1, dim), dtype=jnp.float32)
        self.blocks = tuple(
            TinyTransformerBlock(dim, hidden_dim, key=block_key_i)
            for block_key_i in jr.split(block_key, depth)
        )
        self.norm = eqx.nn.LayerNorm(dim)
        self.head = eqx.nn.Linear(dim, num_classes, key=head_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = _map_tokens(self.patch_embed, x)
        x = jnp.concatenate([self.cls_token, x], axis=0)
        x = x + self.pos_embed[: x.shape[0]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[0]))


class TinyASTLikeEncoder(eqx.Module):
    """Audio-transformer-shaped fixture with class and distillation tokens."""

    dim: int = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)

    patch_embed: TinyPatchEmbed
    cls_token: jax.Array
    dist_token: jax.Array
    pos_embed: jax.Array
    blocks: tuple[TinyTransformerBlock, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(
        self,
        *,
        dim: int = 4,
        hidden_dim: int = 8,
        depth: int = 1,
        patch_features: int = 6,
        num_patches: int = 2,
        num_classes: int = 3,
        key: jax.Array | None = None,
    ):
        key = _key_or_default(key)
        patch_key, block_key, head_key = jr.split(key, 3)
        self.dim = dim
        self.num_patches = num_patches
        self.patch_embed = TinyPatchEmbed(patch_features, dim, key=patch_key)
        self.cls_token = jnp.zeros((1, dim), dtype=jnp.float32)
        self.dist_token = jnp.zeros((1, dim), dtype=jnp.float32)
        self.pos_embed = jnp.zeros((num_patches + 2, dim), dtype=jnp.float32)
        self.blocks = tuple(
            TinyTransformerBlock(dim, hidden_dim, key=block_key_i)
            for block_key_i in jr.split(block_key, depth)
        )
        self.norm = eqx.nn.LayerNorm(dim)
        self.head = eqx.nn.Linear(dim, num_classes, key=head_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = _map_tokens(self.patch_embed, x)
        x = jnp.concatenate([self.cls_token, self.dist_token, x], axis=0)
        x = x + self.pos_embed[: x.shape[0]]
        for block in self.blocks:
            x = block(x)
        pooled = jnp.mean(x[:2], axis=0)
        return self.head(self.norm(pooled))


class TinyConvNeXtBlock(eqx.Module):
    """ConvNeXt-shaped block with depthwise and pointwise parameter paths."""

    dim: int = eqx.field(static=True)

    dwconv: eqx.nn.Conv2d
    norm: eqx.nn.LayerNorm
    pwconv1: eqx.nn.Linear
    pwconv2: eqx.nn.Linear

    def __init__(self, dim: int, hidden_dim: int, *, key: jax.Array):
        dw_key, pw1_key, pw2_key = jr.split(key, 3)
        self.dim = dim
        self.dwconv = eqx.nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, key=dw_key
        )
        self.norm = eqx.nn.LayerNorm(dim)
        self.pwconv1 = eqx.nn.Linear(dim, hidden_dim, key=pw1_key)
        self.pwconv2 = eqx.nn.Linear(hidden_dim, dim, key=pw2_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.dwconv(x)
        channels, height, width = x.shape
        tokens = jnp.moveaxis(x, 0, -1).reshape(height * width, channels)
        tokens = _map_tokens(self.norm, tokens)
        tokens = jax.nn.gelu(_map_tokens(self.pwconv1, tokens))
        tokens = _map_tokens(self.pwconv2, tokens)
        return jnp.moveaxis(tokens.reshape(height, width, channels), -1, 0)


class TinyConvNeXtStage(eqx.Module):
    """Stage wrapper to expose ``stages.*.blocks.*`` paths."""

    blocks: tuple[TinyConvNeXtBlock, ...]

    def __init__(self, dim: int, hidden_dim: int, depth: int, *, key: jax.Array):
        self.blocks = tuple(
            TinyConvNeXtBlock(dim, hidden_dim, key=block_key)
            for block_key in jr.split(key, depth)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        for block in self.blocks:
            x = block(x)
        return x


class TinyConvNeXtLike(eqx.Module):
    """ConvNeXt-shaped fixture with stem, stages, norm, and head paths."""

    dim: int = eqx.field(static=True)

    stem: eqx.nn.Conv2d
    stages: tuple[TinyConvNeXtStage, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(
        self,
        *,
        in_channels: int = 3,
        dim: int = 4,
        hidden_dim: int = 8,
        stage_depths: tuple[int, ...] = (1,),
        num_classes: int = 2,
        key: jax.Array | None = None,
    ):
        key = _key_or_default(key)
        stem_key, stage_key, head_key = jr.split(key, 3)
        self.dim = dim
        self.stem = eqx.nn.Conv2d(
            in_channels, dim, kernel_size=2, stride=2, key=stem_key
        )
        self.stages = tuple(
            TinyConvNeXtStage(dim, hidden_dim, depth, key=stage_key_i)
            for depth, stage_key_i in zip(
                stage_depths, jr.split(stage_key, len(stage_depths))
            )
        )
        self.norm = eqx.nn.LayerNorm(dim)
        self.head = eqx.nn.Linear(dim, num_classes, key=head_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        pooled = jnp.mean(x, axis=(1, 2))
        return self.head(self.norm(pooled))


class TinyTextEncoder(eqx.Module):
    """Text-transformer-shaped fixture with token and position embeddings."""

    dim: int = eqx.field(static=True)
    max_length: int = eqx.field(static=True)

    token_embed: eqx.nn.Embedding
    pos_embed: jax.Array
    blocks: tuple[TinyTransformerBlock, ...]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    def __init__(
        self,
        *,
        vocab_size: int = 5,
        dim: int = 4,
        hidden_dim: int = 8,
        depth: int = 1,
        max_length: int = 3,
        key: jax.Array | None = None,
    ):
        key = _key_or_default(key)
        token_key, block_key, head_key = jr.split(key, 3)
        self.dim = dim
        self.max_length = max_length
        self.token_embed = eqx.nn.Embedding(vocab_size, dim, key=token_key)
        self.pos_embed = jnp.zeros((max_length, dim), dtype=jnp.float32)
        self.blocks = tuple(
            TinyTransformerBlock(dim, hidden_dim, key=block_key_i)
            for block_key_i in jr.split(block_key, depth)
        )
        self.norm = eqx.nn.LayerNorm(dim)
        self.head = eqx.nn.Linear(dim, vocab_size, key=head_key)

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        x = jax.vmap(self.token_embed)(token_ids)
        x = x + self.pos_embed[: x.shape[0]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[0]))


class TinyLinearMLP(eqx.Module):
    """Minimal MLP fixture for callable linear-selector tests."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(
        self,
        *,
        in_features: int = 4,
        hidden_features: int = 5,
        out_features: int = 3,
        key: jax.Array | None = None,
    ):
        key = _key_or_default(key)
        fc1_key, fc2_key = jr.split(key, 2)
        self.fc1 = eqx.nn.Linear(in_features, hidden_features, key=fc1_key)
        self.fc2 = eqx.nn.Linear(hidden_features, out_features, key=fc2_key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x)))


EXPECTED_PARAM_COUNTS = {
    "tiny_ast_like_encoder": 247,
    "tiny_convnext_like": 194,
    "tiny_linear_mlp": 43,
    "tiny_text_encoder": 237,
    "tiny_vision_transformer": 394,
}


def count_params(tree: Any) -> int:
    """Count inexact array scalars in a PyTree."""

    leaves = jtu.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return sum(int(leaf.size) for leaf in leaves if eqx.is_inexact_array(leaf))


def _format_key_path(key_path: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for key in key_path:
        if isinstance(key, jtu.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jtu.SequenceKey):
            parts.append(str(key.idx))
        elif isinstance(key, jtu.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, jtu.FlattenedIndexKey):
            parts.append(str(key.key))
        else:
            parts.append(str(key))
    return ".".join(parts)


def extract_paths(tree: Any) -> tuple[str, ...]:
    """Return dot paths for inexact array leaves in a PyTree."""

    return tuple(
        _format_key_path(key_path)
        for key_path, leaf in jtu.tree_leaves_with_path(tree)
        if eqx.is_inexact_array(leaf)
    )


def assert_tree_allclose(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """Assert that two PyTrees have the same structure and close array leaves."""

    actual_leaves, actual_treedef = jtu.tree_flatten(actual)
    expected_leaves, expected_treedef = jtu.tree_flatten(expected)
    assert actual_treedef == expected_treedef

    for index, (actual_leaf, expected_leaf) in enumerate(
        zip(actual_leaves, expected_leaves, strict=True)
    ):
        if eqx.is_array(actual_leaf) or eqx.is_array(expected_leaf):
            assert jnp.allclose(actual_leaf, expected_leaf, rtol=rtol, atol=atol), index
        else:
            assert actual_leaf == expected_leaf, index


def _param_infos(tree: Any) -> Iterable[ParamInfo]:
    tree = getattr(tree, "param_info", tree)
    return (leaf for leaf in jtu.tree_leaves(tree) if isinstance(leaf, ParamInfo))


def _path_to_str(path: tuple[str | int, ...]) -> str:
    return ".".join(str(part) for part in path)


def assert_no_trainable_with_tag(tree: Any, tag: str) -> None:
    """Assert that no trainable ``ParamInfo`` leaf carries ``tag``."""

    offenders = [
        _path_to_str(info.path)
        for info in _param_infos(tree)
        if info.trainable and tag in info.tags
    ]
    assert not offenders, f"Found trainable leaves tagged {tag!r}: {offenders}"


def assert_only_trainable_tags(tree: Any, tags: Iterable[str]) -> None:
    """Assert that every trainable ``ParamInfo`` leaf has at least one allowed tag."""

    allowed = frozenset(tags)
    offenders = [
        _path_to_str(info.path)
        for info in _param_infos(tree)
        if info.trainable and info.tags.isdisjoint(allowed)
    ]
    assert not offenders, (
        f"Found trainable leaves outside {sorted(allowed)!r}: {offenders}"
    )


__all__ = (
    "EXPECTED_PARAM_COUNTS",
    "TinyASTLikeEncoder",
    "TinyConvNeXtLike",
    "TinyLinearMLP",
    "TinyTextEncoder",
    "TinyVisionTransformer",
    "assert_no_trainable_with_tag",
    "assert_only_trainable_tags",
    "assert_tree_allclose",
    "count_params",
    "extract_paths",
)
