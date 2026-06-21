"""Feature-extraction wrappers for fine-tuning workflows."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax

from ._typing import PyTree
from .heads import IdentityHead, LinearHead
from .pooling import MeanPatchPool, PoolName, pool_features
from .surgery import replace_head


class FeatureExtractor(eqx.Module):
    """Wrap a backbone and return pooled features."""

    model: PyTree
    pool: PoolName | eqx.Module | None = eqx.field(static=True)

    def __init__(self, model: PyTree, *, pool: PoolName | eqx.Module | None = "auto"):
        self.model = model
        self.pool = pool

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

    def __init__(
        self,
        backbone: PyTree,
        head: eqx.Module,
        *,
        pool: PoolName | eqx.Module | None = "auto",
    ):
        self.backbone = backbone
        self.head = head
        self.pool = pool

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

    pool = _prompt_aware_pool(model, pool)
    return pool_features(features, pool, **kwargs)


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
        if key is None:
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


def _call_head(head: eqx.Module, x: Any, *, key: jax.Array | None, inference: bool | None):
    try:
        return head(x, key=key, inference=inference)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        return head(x)


__all__ = (
    "FeatureExtractor",
    "LinearProbe",
    "extract_features",
    "make_linear_probe",
)
