from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .registry import _register_module, _registry_name, _resolve_from_registry

_PREPROCESSOR_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_preprocessor(
    name: str | None = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a tabular preprocessing layer class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        registry_name = _registry_name(cls, name)
        return _register_module(
            _PREPROCESSOR_REGISTRY,
            cls,
            registry_name,
            force,
            "tabular preprocessor",
            add_to_layer_registry=True,
        )

    return decorator


def get_preprocessor(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a tabular preprocessing layer class."""
    return _resolve_from_registry(
        module, _PREPROCESSOR_REGISTRY, "tabular preprocessor"
    )


_NAN_INDICATOR = -2.0
_POSINF_INDICATOR = 2.0
_NEGINF_INDICATOR = 4.0


@register_preprocessor()
class Preprocessor(eqx.Module):
    """Preprocess tabular columns for TabPFN-style feature grouping."""

    feature_group_size: int = eqx.field(static=True)
    use_nan_indicators: bool = eqx.field(static=True)

    def __init__(
        self,
        feature_group_size: int = 3,
        use_nan_indicators: bool = True,
    ) -> None:
        self.feature_group_size = feature_group_size
        self.use_nan_indicators = use_nan_indicators

    def __call__(
        self,
        x: Float[Array, "rows columns"],
        n_train: int,
    ) -> Float[Array, "rows columns grouped_features"]:
        finite = jnp.isfinite(x)
        indicators = None
        if self.use_nan_indicators:
            indicators = (
                jnp.isnan(x) * _NAN_INDICATOR
                + jnp.isposinf(x) * _POSINF_INDICATOR
                + jnp.isneginf(x) * _NEGINF_INDICATOR
            )

        x_train = jnp.where(finite[:n_train], x[:n_train], jnp.nan)
        means = jnp.nan_to_num(jnp.nanmean(x_train, axis=0), nan=0.0)
        x = jnp.where(finite, x, means[None, :])

        train = x[:n_train]
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, ddof=1, keepdims=True)
        std = jnp.where(std == 0, jnp.ones_like(std), std)
        std = jnp.where(n_train == 1, jnp.ones_like(std), std)
        x = jnp.clip((x - mean) / (std + jnp.finfo(std.dtype).eps), min=-100, max=100)

        group_size = self.feature_group_size
        groups = [jnp.roll(x, -(2**i), axis=1) for i in range(group_size)]
        out = jnp.stack(groups, axis=-1)
        if indicators is not None:
            indicator_groups = [
                jnp.roll(indicators, -(2**i), axis=1) for i in range(group_size)
            ]
            out = jnp.concatenate(
                [out, jnp.stack(indicator_groups, axis=-1)],
                axis=-1,
            )
        return out
