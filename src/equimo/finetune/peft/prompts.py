"""Prompt tuning wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .._typing import PyTree


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for visual/soft prompt tuning."""

    num_tokens: int = 10
    depth: Literal["shallow", "deep", "all"] = "deep"
    init: str = "normal"
    init_std: float = 0.02
    prepend_to: Literal["after_cls", "before_all", "input"] = "after_cls"
    prompt_dropout: float = 0.0
    exclude_prompt_tokens_from_pool: bool = True
    train_head: bool = True


@dataclass(frozen=True)
class SoftPromptConfig(PromptConfig):
    """Soft prompt tuning defaults for text encoders."""

    num_tokens: int = 20
    depth: Literal["shallow", "deep", "all"] = "shallow"
    init: str = "from_embedding_if_available"
    prepend_to: Literal["after_cls", "before_all", "input"] = "input"


@dataclass(frozen=True)
class DeepPromptConfig(PromptConfig):
    """Deep prompt / P-tuning v2-style defaults."""

    num_tokens: int = 10
    depth: Literal["shallow", "deep", "all"] = "all"
    share_across_layers: bool = False


@dataclass(frozen=True)
class VPTShallowRecipe(PromptConfig):
    """Visual prompt tuning shallow recipe metadata."""

    num_tokens: int = 50
    depth: Literal["shallow"] = "shallow"
    prompt_dropout: float = 0.0


@dataclass(frozen=True)
class VPTDeepRecipe(PromptConfig):
    """Visual prompt tuning deep recipe metadata."""

    num_tokens: int = 10
    depth: Literal["deep"] = "deep"
    prompt_dropout: float = 0.0


class PromptedModel(eqx.Module):
    """Model wrapper that inserts trainable prompt tokens into feature sequences."""

    base: PyTree
    prompts: tuple[jax.Array, ...]
    config: PromptConfig = eqx.field(static=True)

    @property
    def num_prompt_tokens(self) -> int:
        return self.config.num_tokens

    @property
    def exclude_prompt_tokens_from_pool(self) -> bool:
        return self.config.exclude_prompt_tokens_from_pool

    @property
    def num_base_prefix_tokens(self) -> int:
        return _base_prefix_count(self.base)

    def features(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        **kwargs,
    ) -> jax.Array:
        if _is_equimo_vit_like(self.base):
            return _equimo_vit_features(
                self.base,
                self.prompts,
                self.config,
                *args,
                key=key,
                inference=inference,
                **kwargs,
            )
        if _is_simple_token_model(self.base):
            return _simple_token_features(
                self.base,
                self.prompts,
                self.config,
                *args,
                key=key,
                inference=inference,
                **kwargs,
            )
        raise ValueError(
            "PromptedModel supports ViT-like token models with patch_embed, "
            "prefix tokens, and blocks."
        )

    def __call__(
        self,
        *args,
        key: jax.Array | None = None,
        inference: bool | None = True,
        **kwargs,
    ):
        features = self.features(*args, key=key, inference=inference, **kwargs)
        if hasattr(self.base, "head"):
            if _is_equimo_vit_like(self.base):
                from equimo.utils import pool_sd

                x = jax.vmap(self.base.norm)(features)
                x = pool_sd(
                    x,
                    num_prefix_tokens=self.num_base_prefix_tokens
                    + self.config.num_tokens,
                    pool_type=self.base.global_pool,
                    reduce_include_prefix=False,
                )
                return self.base.head(x)
            token_index = (
                self.config.num_tokens
                if _prepends_before_all(self.config)
                else 0
            )
            return self.base.head(features[token_index])
        return _call_model(self.base, *args, key=key, inference=inference, **kwargs)


def apply_prompts(
    model: PyTree,
    config: PromptConfig | None = None,
    *,
    key: jax.Array,
) -> PromptedModel:
    """Wrap a model with trainable prompt tokens."""

    config = PromptConfig() if config is None else config
    dim = _infer_dim(model)
    prompt_count = _prompt_count(model, config)
    prompts = _init_prompts(model, config, dim, prompt_count, key)
    return PromptedModel(model, prompts, config)


def _prompt_count(model: PyTree, config: PromptConfig) -> int:
    if config.depth == "shallow":
        return 1
    if getattr(config, "share_across_layers", False):
        return 1
    blocks = getattr(model, "blocks", ())
    return max(1, len(blocks))


def _infer_dim(model: PyTree) -> int:
    if hasattr(model, "dim"):
        return int(model.dim)
    if hasattr(model, "pos_embed"):
        return int(model.pos_embed.shape[-1])
    if hasattr(model, "token_embed") and hasattr(model.token_embed, "weight"):
        return int(model.token_embed.weight.shape[-1])
    raise ValueError("Could not infer prompt dimension; pass a model with dim metadata.")


def _init_prompts(
    model: PyTree,
    config: PromptConfig,
    dim: int,
    prompt_count: int,
    key: jax.Array,
) -> tuple[jax.Array, ...]:
    if config.init == "normal":
        return tuple(
            jr.normal(prompt_key, (config.num_tokens, dim), dtype=jnp.float32)
            * config.init_std
            for prompt_key in jr.split(key, prompt_count)
        )
    if config.init == "from_embedding_if_available":
        prompt = _prompt_from_embedding(model, config.num_tokens)
        if prompt is not None:
            return tuple(prompt for _ in range(prompt_count))
        return tuple(
            jr.normal(prompt_key, (config.num_tokens, dim), dtype=jnp.float32)
            * config.init_std
            for prompt_key in jr.split(key, prompt_count)
        )
    raise ValueError(
        "Unsupported prompt init "
        f"{config.init!r}; expected normal or from_embedding_if_available."
    )


def _prompt_from_embedding(model: PyTree, num_tokens: int) -> jax.Array | None:
    token_embed = getattr(model, "token_embed", None)
    weight = getattr(token_embed, "weight", None)
    if weight is None:
        return None
    indices = jnp.arange(num_tokens) % weight.shape[0]
    return weight[indices]


def _equimo_vit_features(
    model,
    prompts: tuple[jax.Array, ...],
    config: PromptConfig,
    x: jax.Array,
    *,
    key: jax.Array | None,
    inference: bool | None,
    **kwargs,
) -> jax.Array:
    key = jr.PRNGKey(0) if key is None else key
    key_pos, *block_subkeys = jr.split(key, len(model.blocks) + 1)
    mask = kwargs.pop("mask", None)
    x = model.patch_embed(x)

    if mask is not None:
        if model.mask_token is None:
            raise AssertionError(
                "To use masked forward, init the model with `use_mask_token=True`."
            )
        if model.dynamic_img_size:
            mask = jnp.expand_dims(mask, axis=0)
            value = jnp.reshape(model.mask_token, (-1, 1, 1))
        else:
            mask = jnp.reshape(mask, (-1, 1))
            value = model.mask_token
        x = jnp.where(mask, x, value.astype(x.dtype))

    if model.local_pos_embed is not None:
        if model.dynamic_img_size:
            _, height, width = x.shape
        else:
            height = width = model.embed_size

    if model.global_pos_embed is not None:
        x = model.global_pos_embed(
            x,
            cls_token=model.cls_token,
            reg_tokens=model.reg_tokens,
            dynamic_img_size=model.dynamic_img_size,
        )
    else:
        prefix = [token for token in (model.cls_token, model.reg_tokens) if token is not None]
        if model.dynamic_img_size:
            x = jnp.moveaxis(x, 0, -1).reshape((-1, x.shape[0]))
        x = jnp.concatenate([*prefix, x], axis=0) if prefix else x

    rope_sincos = None
    if model.local_pos_embed is not None and inference:
        rope_sincos = model.local_pos_embed.get_sincos(
            H=height,
            W=width,
            inference=inference,
            key=key_pos,
        )

    x = _run_prompted_blocks(
        model.blocks,
        x,
        prompts,
        config,
        block_subkeys,
        inference=inference,
        rope_sincos=rope_sincos,
        key_pos=key_pos,
        model=model,
        **kwargs,
    )
    return x


def _simple_token_features(
    model,
    prompts: tuple[jax.Array, ...],
    config: PromptConfig,
    x: jax.Array,
    *,
    key: jax.Array | None,
    inference: bool | None,
    **kwargs,
) -> jax.Array:
    del kwargs
    x = _embed_input_tokens(model, x)
    prefix = _simple_prefix_tokens(model)
    x = jnp.concatenate([*prefix, x], axis=0) if prefix else x
    if hasattr(model, "pos_embed"):
        x = x + model.pos_embed[: x.shape[0]]
    block_keys = jr.split(key, len(model.blocks)) if key is not None else (None,) * len(model.blocks)
    x = _run_prompted_blocks(
        model.blocks,
        x,
        prompts,
        config,
        block_keys,
        inference=inference,
    )
    return _map_tokens(model.norm, x) if hasattr(model, "norm") else x


def _run_prompted_blocks(
    blocks,
    x: jax.Array,
    prompts: tuple[jax.Array, ...],
    config: PromptConfig,
    block_keys,
    *,
    inference: bool | None,
    rope_sincos=None,
    key_pos: jax.Array | None = None,
    model=None,
    **kwargs,
) -> jax.Array:
    if not blocks:
        return _insert_prompt(x, _prompt_for_layer(prompts, config, 0, x, None), config)

    if config.depth == "shallow":
        prompt = _prompt_for_layer(prompts, config, 0, x, None)
        x = _insert_prompt(x, prompt, config)
        rope_sincos = _insert_prompt_rope(rope_sincos, prompt, config)

    for index, (block, block_key) in enumerate(zip(blocks, block_keys, strict=True)):
        prompt = None
        if _uses_deep_prompts(config):
            prompt = _prompt_for_layer(prompts, config, index, x, block_key)
            if index == 0:
                x = _insert_prompt(x, prompt, config)
                rope_sincos = _insert_prompt_rope(rope_sincos, prompt, config)
            else:
                x = _replace_prompt(x, prompt, config)
        if (
            model is not None
            and getattr(model, "local_pos_embed", None) is not None
            and not inference
            and key_pos is not None
        ):
            key_pos, key_rope = jr.split(key_pos, 2)
            rope_sincos = model.local_pos_embed.get_sincos(
                H=model.embed_size,
                W=model.embed_size,
                inference=inference,
                key=key_rope,
            )
            if config.depth == "shallow":
                prompt = _prompt_for_layer(prompts, config, 0, x, None)
            if prompt is not None:
                rope_sincos = _insert_prompt_rope(rope_sincos, prompt, config)
        block_kwargs = dict(kwargs)
        if rope_sincos is not None:
            block_kwargs["rope_sincos"] = rope_sincos
        x = _call_block(
            block,
            x,
            key=block_key,
            inference=inference,
            **block_kwargs,
        )
    return x


def _prompt_for_layer(
    prompts: tuple[jax.Array, ...],
    config: PromptConfig,
    index: int,
    x: jax.Array,
    key: jax.Array | None,
) -> jax.Array:
    prompt = prompts[0 if config.depth == "shallow" else min(index, len(prompts) - 1)]
    prompt = prompt.astype(x.dtype)
    if config.prompt_dropout > 0.0 and key is not None:
        prompt_key = key
    else:
        prompt_key = None
    if config.prompt_dropout > 0.0 and prompt_key is not None:
        if key is None:
            raise ValueError("A PRNG key is required when prompt dropout is active.")
        prompt = _dropout(prompt, config.prompt_dropout, prompt_key)
    return prompt


def _insert_prompt(x: jax.Array, prompt: jax.Array, config: PromptConfig) -> jax.Array:
    if _prepends_before_all(config):
        return jnp.concatenate([prompt, x], axis=0)
    return jnp.concatenate([x[:1], prompt, x[1:]], axis=0)


def _insert_prompt_rope(
    rope_sincos: tuple[jax.Array, jax.Array] | None,
    prompt: jax.Array,
    config: PromptConfig,
) -> tuple[jax.Array, jax.Array] | None:
    if rope_sincos is None:
        return None

    sin, cos = rope_sincos
    prompt_rows = prompt.shape[0]
    insert_at = 0 if _prepends_before_all(config) else 1
    sin_prompt = jnp.zeros((prompt_rows, sin.shape[-1]), dtype=sin.dtype)
    cos_prompt = jnp.ones((prompt_rows, cos.shape[-1]), dtype=cos.dtype)
    sin = jnp.concatenate([sin[:insert_at], sin_prompt, sin[insert_at:]], axis=0)
    cos = jnp.concatenate([cos[:insert_at], cos_prompt, cos[insert_at:]], axis=0)
    return sin, cos


def _replace_prompt(x: jax.Array, prompt: jax.Array, config: PromptConfig) -> jax.Array:
    n = prompt.shape[0]
    if x.shape[0] >= n and _prepends_before_all(config):
        return jnp.concatenate([prompt, x[n:]], axis=0)
    if x.shape[0] >= n + 1 and config.prepend_to == "after_cls":
        return jnp.concatenate([x[:1], prompt, x[1 + n :]], axis=0)
    return _insert_prompt(x, prompt, config)


def _call_block(block, x: jax.Array, *, key, inference, **kwargs) -> jax.Array:
    call_kwargs = dict(kwargs)
    if key is not None:
        call_kwargs["key"] = key
    if inference is not None:
        call_kwargs["inference"] = inference
    try:
        return block(x, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        if key is None:
            call_kwargs.pop("key", None)
        try:
            return block(x, **call_kwargs)
        except TypeError as second_error:
            if "unexpected keyword argument" not in str(second_error):
                raise
            return block(x)


def _is_equimo_vit_like(model) -> bool:
    return all(
        hasattr(model, name)
        for name in (
            "patch_embed",
            "blocks",
            "global_pos_embed",
            "local_pos_embed",
            "num_prefix_tokens",
            "global_pool",
        )
    )


def _is_simple_token_model(model) -> bool:
    return (
        hasattr(model, "blocks")
        and (hasattr(model, "patch_embed") or hasattr(model, "token_embed"))
    )


def _embed_input_tokens(model, x: jax.Array) -> jax.Array:
    if hasattr(model, "token_embed"):
        return jax.vmap(model.token_embed)(x)
    return _map_tokens(model.patch_embed, x)


def _simple_prefix_tokens(model) -> list[jax.Array]:
    tokens: list[jax.Array] = []
    for name in ("cls_token", "dist_token", "reg_tokens"):
        token = getattr(model, name, None)
        if token is not None:
            tokens.append(token)
    return tokens


def _base_prefix_count(model) -> int:
    if hasattr(model, "num_prefix_tokens"):
        return int(model.num_prefix_tokens)
    return sum(token.shape[0] for token in _simple_prefix_tokens(model))


def _map_tokens(fn, x: jax.Array) -> jax.Array:
    return fn(x) if x.ndim == 1 else jax.vmap(fn)(x)


def _call_features(model, *args, key, inference, **kwargs):
    if not hasattr(model, "features"):
        raise ValueError("PromptedModel requires the base model to expose features().")
    return _call_with_optional_key(model.features, *args, key=key, inference=inference, **kwargs)


def _call_model(model, *args, key, inference, **kwargs):
    return _call_with_optional_key(model, *args, key=key, inference=inference, **kwargs)


def _call_with_optional_key(fn, *args, key, inference, **kwargs):
    call_kwargs = dict(kwargs)
    if key is not None:
        call_kwargs["key"] = key
    if inference is not None:
        call_kwargs["inference"] = inference
    try:
        return fn(*args, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        try:
            return fn(*args, **call_kwargs)
        except TypeError as second_error:
            if "unexpected keyword argument" not in str(second_error):
                raise
            call_kwargs.pop("key", None)
            return fn(*args, **call_kwargs)


def _dropout(x: jax.Array, rate: float, key: jax.Array) -> jax.Array:
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def _prepends_before_all(config: PromptConfig) -> bool:
    return config.prepend_to in {"before_all", "input"}


def _uses_deep_prompts(config: PromptConfig) -> bool:
    return config.depth in {"deep", "all"}


__all__ = (
    "DeepPromptConfig",
    "PromptConfig",
    "PromptedModel",
    "SoftPromptConfig",
    "VPTDeepRecipe",
    "VPTShallowRecipe",
    "apply_prompts",
)
