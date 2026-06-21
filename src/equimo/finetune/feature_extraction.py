"""Feature-extraction wrappers for fine-tuning workflows."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax

from ._typing import PyTree
from .config import FeatureSpec
from .heads import IdentityHead, LinearHead
from .pooling import GlobalAveragePool, MeanPatchPool, MeanTokenPool, PoolName, pool_features
from .surgery import replace_head


class FeatureExtractor(eqx.Module):
    """Wrap a backbone and return pooled features."""

    model: PyTree
    pool: PoolName | eqx.Module | None = eqx.field(static=True)
    feature_spec: FeatureSpec | None = eqx.field(static=True)

    def __init__(
        self,
        model: PyTree,
        *,
        pool: PoolName | eqx.Module | None = "auto",
        feature_spec: FeatureSpec | None = None,
    ):
        self.model = model
        self.pool = pool
        self.feature_spec = feature_spec

    def __call__(self, *args, key: jax.Array | None = None, inference: bool | None = True, **kwargs):
        return extract_features(
            self.model,
            *args,
            pool=self.pool,
            key=key,
            inference=inference,
            **kwargs,
        )


class LinearProbe(eqx.Module):
    """Backbone feature extractor plus a trainable task head."""

    backbone: PyTree
    head: eqx.Module
    pool: PoolName | eqx.Module | None = eqx.field(static=True)
    feature_spec: FeatureSpec | None = eqx.field(static=True)

    def __init__(
        self,
        backbone: PyTree,
        head: eqx.Module,
        *,
        pool: PoolName | eqx.Module | None = "auto",
        feature_spec: FeatureSpec | None = None,
    ):
        self.backbone = backbone
        self.head = head
        self.pool = pool
        self.feature_spec = feature_spec

    def __call__(self, *args, key: jax.Array | None = None, inference: bool | None = True, **kwargs):
        features = extract_features(
            self.backbone,
            *args,
            pool=self.pool,
            key=key,
            inference=inference,
            **kwargs,
        )
        return _call_head(self.head, features, key=key, inference=inference)


class HeadOnlyModel(LinearProbe):
    """Backbone feature extractor plus a trainable replacement head."""


def extract_features(
    model: PyTree,
    *args,
    pool: PoolName | eqx.Module | None = "auto",
    key: jax.Array | None = None,
    inference: bool | None = True,
    **kwargs,
) -> Any:
    """Call a model feature path and apply an optional pooling policy."""

    if hasattr(model, "features"):
        features = _call_with_optional_key(
            model.features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )
    elif hasattr(model, "forward_features"):
        features = _call_with_optional_key(
            model.forward_features,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )
    else:
        features = _call_with_optional_key(
            model,
            *args,
            key=key,
            inference=inference,
            **kwargs,
        )

    pool = _resolve_pool(model, features, _prompt_aware_pool(model, pool))
    return pool_features(features, pool, key=key, **kwargs)


def make_linear_probe(
    backbone: PyTree,
    *,
    in_features: int,
    out_features: int,
    key: jax.Array,
    pool: PoolName | eqx.Module | None = "auto",
    head: eqx.Module | None = None,
) -> LinearProbe:
    """Build a linear-probe wrapper with an identity backbone head."""

    probe_head = (
        LinearHead(in_features, out_features, key=key)
        if head is None
        else head
    )
    try:
        backbone = replace_head(backbone, IdentityHead())
    except ValueError:
        pass
    return LinearProbe(backbone, probe_head, pool=pool)


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


def _prompt_aware_pool(model: PyTree, pool: PoolName | eqx.Module | None):
    if (
        pool == "mean_patch"
        and getattr(model, "exclude_prompt_tokens_from_pool", False)
        and getattr(model, "num_prompt_tokens", 0)
    ):
        return MeanPatchPool(
            num_prefix_tokens=int(getattr(model, "num_base_prefix_tokens", 1)),
            num_prompt_tokens=int(model.num_prompt_tokens),
        )
    return pool


def _resolve_pool(
    model: PyTree,
    features: Any,
    pool: PoolName | eqx.Module | None,
) -> PoolName | eqx.Module | None:
    if pool != "auto":
        return pool
    if isinstance(features, dict):
        return "auto"
    if not eqx.is_array(features):
        return pool

    model_name = model.__class__.__name__.lower()
    if _is_audio_model(model, model_name):
        return "mean_frame"
    if _is_text_model(model):
        if hasattr(model, "pooler") or hasattr(model, "cls_token"):
            return "cls"
        return MeanTokenPool()
    if _is_convnet_model(model):
        return GlobalAveragePool()
    if _is_mae_like(model, model_name):
        return _mean_patch_pool_for_model(model)
    if _is_vit_like(model):
        if getattr(model, "global_pool", None) in {"avg", "mean", "mean_patch"}:
            return _mean_patch_pool_for_model(model)
        return "cls"
    if features.ndim <= 1:
        return "none"
    return "cls"


def _is_vit_like(model: PyTree) -> bool:
    return hasattr(model, "patch_embed") and hasattr(model, "blocks")


def _is_mae_like(model: PyTree, model_name: str) -> bool:
    return "mae" in model_name or getattr(model, "pool_policy", None) == "mean_patch"


def _is_audio_model(model: PyTree, model_name: str) -> bool:
    return (
        "ast" in model_name
        or "audio" in model_name
        or hasattr(model, "dist_token")
        or getattr(model, "modality", None) == "audio"
    )


def _is_text_model(model: PyTree) -> bool:
    return (
        hasattr(model, "token_embed")
        or hasattr(model, "token_embedding")
        or getattr(model, "modality", None) == "text"
    )


def _is_convnet_model(model: PyTree) -> bool:
    return (
        hasattr(model, "stem")
        and hasattr(model, "stages")
        and not hasattr(model, "patch_embed")
    )


def _mean_patch_pool_for_model(model: PyTree) -> MeanPatchPool:
    return MeanPatchPool(
        num_prefix_tokens=int(getattr(model, "num_prefix_tokens", _base_prefix_count(model))),
        num_prompt_tokens=int(getattr(model, "num_prompt_tokens", 0)),
    )


def _base_prefix_count(model: PyTree) -> int:
    count = 0
    for name in ("cls_token", "dist_token"):
        token = getattr(model, name, None)
        if token is not None:
            count += int(token.shape[0]) if hasattr(token, "shape") and token.ndim > 1 else 1
    reg_tokens = getattr(model, "reg_tokens", None)
    if reg_tokens is not None:
        count += int(reg_tokens.shape[0]) if hasattr(reg_tokens, "shape") else 1
    return count


def _call_head(head: eqx.Module, x: Any, *, key: jax.Array | None, inference: bool | None):
    try:
        return head(x, key=key, inference=inference)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        return head(x)


__all__ = (
    "FeatureExtractor",
    "HeadOnlyModel",
    "LinearProbe",
    "extract_features",
    "make_linear_probe",
)
