"""Bottleneck adapter modules and model surgery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..config import FineTuneBundle, TargetSpec
from ..heads import ActivationName
from ..paths import key_path_to_path, path_to_str, str_to_path
from ..selectors import resolve_target
from ..tags import Tagger, canonical_tags_for_path
from .base import get_path
from .lora import architecture_hash


AdapterPlacement = Literal["after_mlp", "parallel", "both"]


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for bottleneck adapters."""

    bottleneck: int | None = None
    reduction_factor: int = 16
    bottleneck_min: int = 16
    bottleneck_max: int = 64
    placement: AdapterPlacement = "after_mlp"
    activation: ActivationName = "gelu"
    dropout: float = 0.0
    down_init: str = "kaiming_uniform"
    up_init: str = "zeros"
    target: TargetSpec = TargetSpec(tags=("block",))


@dataclass(frozen=True)
class AdaptFormerConfig:
    """Configuration for AdaptFormer-style parallel adapters."""

    bottleneck: int = 64
    placement: Literal["parallel_mlp"] = "parallel_mlp"
    activation: ActivationName = "gelu"
    dropout: float = 0.0
    up_init: str = "zeros"
    scale_init: float = 1.0
    target: TargetSpec = TargetSpec(tags=("block",))


class BottleneckAdapter(eqx.Module):
    """Residual bottleneck adapter with identity-preserving zero-up init."""

    down: eqx.nn.Linear
    up: eqx.nn.Linear
    activation: ActivationName = eqx.field(static=True)
    dropout: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        *,
        key: jax.Array,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        down_init: str = "kaiming_uniform",
        up_init: str = "zeros",
        down: eqx.nn.Linear | None = None,
        up: eqx.nn.Linear | None = None,
    ):
        key_down, key_up = jr.split(key, 2)
        self.down = (
            _init_linear(
                eqx.nn.Linear(dim, bottleneck, key=key_down),
                key_down,
                down_init,
            )
            if down is None
            else down
        )
        self.up = (
            _init_linear(
                eqx.nn.Linear(bottleneck, dim, key=key_up),
                key_up,
                up_init,
            )
            if up is None
            else up
        )
        self.activation = activation
        self.dropout = dropout

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        y = _apply_last_axis(self.down, x)
        y = _activation(self.activation)(y)
        if self.dropout > 0.0 and not inference:
            if key is None:
                raise ValueError("A PRNG key is required when adapter dropout is active.")
            y = _dropout(y, self.dropout, key)
        return _apply_last_axis(self.up, y)


class AdaptFormerAdapter(eqx.Module):
    """AdaptFormer adapter branch."""

    adapter: BottleneckAdapter
    scale: jax.Array

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        *,
        key: jax.Array,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        up_init: str = "zeros",
        scale_init: float = 1.0,
        adapter: BottleneckAdapter | None = None,
        scale: jax.Array | None = None,
    ):
        self.adapter = (
            BottleneckAdapter(
                dim,
                bottleneck,
                key=key,
                activation=activation,
                dropout=dropout,
                up_init=up_init,
            )
            if adapter is None
            else adapter
        )
        self.scale = jnp.asarray(scale_init, dtype=jnp.float32) if scale is None else scale

    def __call__(
        self,
        x: jax.Array,
        *,
        key: jax.Array | None = None,
        inference: bool | None = True,
    ) -> jax.Array:
        return self.adapter(x, key=key, inference=inference) * self.scale


class SerialAdapterBlock(eqx.Module):
    """Wrap a block and add residual adapters after the block output."""

    base: eqx.Module
    adapters: tuple[BottleneckAdapter, ...]
    adapter_names: tuple[str, ...] = eqx.field(static=True, default=())
    active_adapter: str | None = eqx.field(static=True, default=None)

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        y = _call_base(self.base, x, *args, key=key, inference=inference, **kwargs)
        adapters = _active_adapters(self)
        keys = _split_optional_key(key, len(adapters))
        for adapter, adapter_key in zip(adapters, keys, strict=True):
            y = y + adapter(y, key=adapter_key, inference=inference)
        return y


class ParallelAdapterBlock(eqx.Module):
    """Wrap a block and add an adapter branch from the block input."""

    base: eqx.Module
    adapter: BottleneckAdapter

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        key_base, key_adapter = _split_pair(key)
        y = _call_base(self.base, x, *args, key=key_base, inference=inference, **kwargs)
        return y + self.adapter(x, key=key_adapter, inference=inference)


class AdaptFormerBlock(eqx.Module):
    """Wrap a block with an AdaptFormer-style parallel branch."""

    base: eqx.Module
    adapter: AdaptFormerAdapter

    def __call__(self, x: jax.Array, *args, key: jax.Array | None = None, inference: bool | None = None, **kwargs):
        key_base, key_adapter = _split_pair(key)
        y = _call_base(self.base, x, *args, key=key_base, inference=inference, **kwargs)
        return y + self.adapter(x, key=key_adapter, inference=inference)


def apply_adapters(
    model: PyTree,
    config: AdapterConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Insert bottleneck adapters into selected blocks."""

    config = AdapterConfig() if config is None else config
    block_paths = _target_block_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(block_paths))
    updated = model

    for block_path, subkey in zip(block_paths, keys, strict=True):
        block = get_path(updated, block_path)
        if isinstance(block, (SerialAdapterBlock, ParallelAdapterBlock, AdaptFormerBlock)):
            continue
        dim = _infer_dim(block)
        bottleneck = _resolve_bottleneck(dim, config)
        adapter_keys = jr.split(subkey, 2 if config.placement == "both" else 1)
        if config.placement == "parallel":
            wrapper = ParallelAdapterBlock(
                block,
                BottleneckAdapter(
                    dim,
                    bottleneck,
                    key=adapter_keys[0],
                    activation=config.activation,
                    dropout=config.dropout,
                    down_init=config.down_init,
                    up_init=config.up_init,
                ),
            )
        else:
            adapters = tuple(
                BottleneckAdapter(
                    dim,
                    bottleneck,
                    key=adapter_key,
                    activation=config.activation,
                    dropout=config.dropout,
                    down_init=config.down_init,
                    up_init=config.up_init,
                )
                for adapter_key in adapter_keys
            )
            wrapper = SerialAdapterBlock(block, adapters)
        updated = eqx.tree_at(lambda tree, p=block_path: get_path(tree, p), updated, wrapper)

    return updated


def add_adapter(
    model: PyTree,
    *,
    name: str,
    config: AdapterConfig | None = None,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Add a named serial adapter bank entry and keep the current active entry."""

    if not name:
        raise ValueError("Adapter name must be a non-empty string.")
    config = AdapterConfig() if config is None else config
    if config.placement == "parallel":
        raise ValueError("Named adapter banks currently support 'after_mlp' and 'both'.")

    block_paths = _target_block_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(block_paths))
    updated = model
    for block_path, subkey in zip(block_paths, keys, strict=True):
        block = get_path(updated, block_path)
        if isinstance(block, ParallelAdapterBlock):
            raise ValueError("Named adapter banks do not support parallel adapter wrappers.")

        base = block.base if isinstance(block, SerialAdapterBlock) else block
        existing_adapters = block.adapters if isinstance(block, SerialAdapterBlock) else ()
        existing_names = (
            block.adapter_names
            if isinstance(block, SerialAdapterBlock) and block.adapter_names
            else ("default",) * len(existing_adapters)
        )
        active = (
            block.active_adapter
            if isinstance(block, SerialAdapterBlock) and block.active_adapter is not None
            else ("default" if existing_names else name)
        )
        if name in existing_names:
            raise ValueError(f"Adapter {name!r} already exists at {path_to_str(block_path)}.")

        dim = _infer_dim(base)
        bottleneck = _resolve_bottleneck(dim, config)
        adapter_count = 2 if config.placement == "both" else 1
        new_adapters = tuple(
            BottleneckAdapter(
                dim,
                bottleneck,
                key=adapter_key,
                activation=config.activation,
                dropout=config.dropout,
                down_init=config.down_init,
                up_init=config.up_init,
            )
            for adapter_key in jr.split(subkey, adapter_count)
        )
        wrapper = SerialAdapterBlock(
            base,
            (*existing_adapters, *new_adapters),
            (*existing_names, *((name,) * len(new_adapters))),
            active,
        )
        updated = eqx.tree_at(lambda tree, p=block_path: get_path(tree, p), updated, wrapper)
    return updated


def set_active_adapter(model: PyTree, name: str) -> PyTree:
    """Return ``model`` with all named serial adapter banks switched to ``name``."""

    wrappers = iter_adapter_wrappers(model)
    named_wrappers = [
        (path, wrapper)
        for path, wrapper in wrappers
        if isinstance(wrapper, SerialAdapterBlock) and wrapper.adapter_names
    ]
    if not named_wrappers:
        raise ValueError("No named adapter banks found.")

    updated = model
    for path, wrapper in named_wrappers:
        if name not in wrapper.adapter_names:
            raise ValueError(f"Adapter {name!r} not found at {path_to_str(path)}.")
        updated_wrapper = SerialAdapterBlock(
            wrapper.base,
            wrapper.adapters,
            wrapper.adapter_names,
            name,
        )
        updated = eqx.tree_at(
            lambda tree, p=path: get_path(tree, p),
            updated,
            updated_wrapper,
        )
    return updated


def apply_adaptformer(
    model: PyTree,
    config: AdaptFormerConfig | None = None,
    *,
    key: jax.Array,
    tagger: Tagger = canonical_tags_for_path,
) -> PyTree:
    """Insert AdaptFormer-style adapters into selected blocks."""

    config = AdaptFormerConfig() if config is None else config
    block_paths = _target_block_paths(model, config.target, tagger=tagger)
    keys = jr.split(key, len(block_paths))
    updated = model

    for block_path, subkey in zip(block_paths, keys, strict=True):
        block = get_path(updated, block_path)
        if isinstance(block, AdaptFormerBlock):
            continue
        dim = _infer_dim(block)
        wrapper = AdaptFormerBlock(
            block,
            AdaptFormerAdapter(
                dim,
                config.bottleneck,
                key=subkey,
                activation=config.activation,
                dropout=config.dropout,
                up_init=config.up_init,
                scale_init=config.scale_init,
            ),
        )
        updated = eqx.tree_at(lambda tree, p=block_path: get_path(tree, p), updated, wrapper)

    return updated


def extract_adapter_delta(model: PyTree) -> FineTuneBundle:
    """Extract adapter wrapper state into a delta bundle."""

    entries = []
    for path, wrapper in iter_adapter_wrappers(model):
        entries.append(_adapter_entry(path, wrapper))

    return FineTuneBundle(
        method="adapter",
        schema_version=1,
        architecture_hash=architecture_hash(strip_adapters(model)),
        adapter_config={"entries": entries},
    )


def load_adapter_delta(base_model: PyTree, bundle: FineTuneBundle) -> PyTree:
    """Load adapter deltas into a compatible base model."""

    if bundle.method != "adapter":
        raise ValueError(f"Expected an adapter bundle, got method={bundle.method!r}.")
    actual_hash = architecture_hash(base_model)
    if bundle.architecture_hash and bundle.architecture_hash != actual_hash:
        raise ValueError(
            "Adapter delta architecture hash mismatch: "
            f"expected {bundle.architecture_hash}, got {actual_hash}."
        )

    updated = base_model
    for entry in bundle.adapter_config.get("entries", ()):
        path = str_to_path(entry["path"])
        block = get_path(updated, path)
        wrapper = _wrapper_from_entry(block, entry)
        updated = eqx.tree_at(lambda tree, p=path: get_path(tree, p), updated, wrapper)
    return updated


def strip_adapters(model: PyTree) -> PyTree:
    """Replace adapter wrappers by their base blocks."""

    stripped = model
    for path, wrapper in iter_adapter_wrappers(stripped):
        stripped = eqx.tree_at(lambda tree, p=path: get_path(tree, p), stripped, wrapper.base)
    return stripped


def iter_adapter_wrappers(
    model: PyTree,
) -> tuple[tuple[Path, SerialAdapterBlock | ParallelAdapterBlock | AdaptFormerBlock], ...]:
    """Return path/wrapper pairs for adapter-wrapped blocks."""

    wrapper_types = (SerialAdapterBlock, ParallelAdapterBlock, AdaptFormerBlock)
    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, wrapper_types),
        )
        if isinstance(leaf, wrapper_types)
    )


def _adapter_entry(
    path: Path,
    wrapper: SerialAdapterBlock | ParallelAdapterBlock | AdaptFormerBlock,
) -> dict[str, Any]:
    if isinstance(wrapper, SerialAdapterBlock):
        return {
            "path": path_to_str(path),
            "class": "SerialAdapterBlock",
            "adapters": [_bottleneck_state(adapter) for adapter in wrapper.adapters],
            "adapter_names": wrapper.adapter_names,
            "active_adapter": wrapper.active_adapter,
        }
    if isinstance(wrapper, ParallelAdapterBlock):
        return {
            "path": path_to_str(path),
            "class": "ParallelAdapterBlock",
            "adapter": _bottleneck_state(wrapper.adapter),
        }
    return {
        "path": path_to_str(path),
        "class": "AdaptFormerBlock",
        "adapter": _adaptformer_state(wrapper.adapter),
    }


def _wrapper_from_entry(block: eqx.Module, entry: dict[str, Any]):
    if entry["class"] == "SerialAdapterBlock":
        return SerialAdapterBlock(
            block,
            tuple(_bottleneck_from_state(state) for state in entry["adapters"]),
            tuple(entry.get("adapter_names", ())),
            entry.get("active_adapter"),
        )
    if entry["class"] == "ParallelAdapterBlock":
        return ParallelAdapterBlock(block, _bottleneck_from_state(entry["adapter"]))
    if entry["class"] == "AdaptFormerBlock":
        return AdaptFormerBlock(block, _adaptformer_from_state(entry["adapter"]))
    raise ValueError(f"Unknown adapter wrapper class {entry['class']!r}.")


def _bottleneck_state(adapter: BottleneckAdapter) -> dict[str, Any]:
    return {
        "dim": adapter.down.in_features,
        "bottleneck": adapter.down.out_features,
        "activation": adapter.activation,
        "dropout": adapter.dropout,
        "down": _linear_state(adapter.down),
        "up": _linear_state(adapter.up),
    }


def _adaptformer_state(adapter: AdaptFormerAdapter) -> dict[str, Any]:
    state = _bottleneck_state(adapter.adapter)
    state["scale"] = adapter.scale
    return state


def _bottleneck_from_state(state: dict[str, Any]) -> BottleneckAdapter:
    return BottleneckAdapter(
        int(state["dim"]),
        int(state["bottleneck"]),
        key=jr.PRNGKey(0),
        activation=state["activation"],
        dropout=float(state["dropout"]),
        down=_linear_from_state(state["down"]),
        up=_linear_from_state(state["up"]),
    )


def _adaptformer_from_state(state: dict[str, Any]) -> AdaptFormerAdapter:
    adapter = _bottleneck_from_state(state)
    return AdaptFormerAdapter(
        int(state["dim"]),
        int(state["bottleneck"]),
        key=jr.PRNGKey(0),
        activation=state["activation"],
        dropout=float(state["dropout"]),
        adapter=adapter,
        scale=state["scale"],
    )


def _linear_state(linear: eqx.nn.Linear) -> dict[str, Any]:
    return {
        "in_features": linear.in_features,
        "out_features": linear.out_features,
        "use_bias": linear.bias is not None,
        "weight": linear.weight,
        "bias": linear.bias,
    }


def _linear_from_state(state: dict[str, Any]) -> eqx.nn.Linear:
    linear = eqx.nn.Linear(
        int(state["in_features"]),
        int(state["out_features"]),
        use_bias=bool(state["use_bias"]),
        key=jr.PRNGKey(0),
    )
    linear = eqx.tree_at(lambda layer: layer.weight, linear, state["weight"])
    if state["bias"] is not None:
        linear = eqx.tree_at(lambda layer: layer.bias, linear, state["bias"])
    return linear


def _target_block_paths(
    model: PyTree,
    target: TargetSpec,
    *,
    tagger: Tagger,
) -> tuple[Path, ...]:
    block_paths = {_block_path(info.path) for info in resolve_target(model, target, tagger=tagger)}
    block_paths.discard(())
    return tuple(sorted(block_paths, key=path_to_str))


def _block_path(path: Path) -> Path:
    parts = tuple(str(part) for part in path)
    for index, part in enumerate(parts[:-1]):
        if part in {"blocks", "block"}:
            return path[: index + 2]
    return path[:-1]


def _infer_dim(block: eqx.Module) -> int:
    for leaf in jtu.tree_leaves(block, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)):
        if isinstance(leaf, eqx.nn.Linear):
            return int(leaf.in_features)
    raise ValueError(f"Could not infer adapter feature dimension for {type(block).__name__}.")


def _resolve_bottleneck(dim: int, config: AdapterConfig) -> int:
    if config.bottleneck is not None:
        return config.bottleneck
    bottleneck = max(1, dim // config.reduction_factor)
    return min(max(bottleneck, config.bottleneck_min), config.bottleneck_max)


def _active_adapters(wrapper: SerialAdapterBlock) -> tuple[BottleneckAdapter, ...]:
    if not wrapper.adapter_names or wrapper.active_adapter is None:
        return wrapper.adapters
    return tuple(
        adapter
        for adapter, name in zip(wrapper.adapters, wrapper.adapter_names, strict=True)
        if name == wrapper.active_adapter
    )


def _init_linear(linear: eqx.nn.Linear, key: jax.Array, init: str) -> eqx.nn.Linear:
    if init == "zeros":
        weight = jnp.zeros_like(linear.weight)
    elif init == "kaiming_uniform":
        bound = jnp.sqrt(6.0 / linear.in_features)
        weight = jr.uniform(
            key,
            linear.weight.shape,
            minval=-bound,
            maxval=bound,
            dtype=linear.weight.dtype,
        )
    else:
        raise ValueError(f"Unsupported adapter init {init!r}.")
    linear = eqx.tree_at(lambda m: m.weight, linear, weight)
    if linear.bias is not None:
        linear = eqx.tree_at(lambda m: m.bias, linear, jnp.zeros_like(linear.bias))
    return linear


def _apply_last_axis(module: eqx.nn.Linear, x: jax.Array) -> jax.Array:
    if x.ndim == 1:
        return module(x)
    leading_shape = x.shape[:-1]
    x_flat = x.reshape((-1, x.shape[-1]))
    y_flat = jax.vmap(module)(x_flat)
    return y_flat.reshape((*leading_shape, y_flat.shape[-1]))


def _activation(name: ActivationName):
    if name == "gelu":
        return jax.nn.gelu
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    if name == "tanh":
        return jnp.tanh
    if name == "identity":
        return lambda x: x
    raise ValueError(f"Unsupported adapter activation {name!r}.")


def _dropout(x: jax.Array, rate: float, key: jax.Array) -> jax.Array:
    keep_prob = 1.0 - rate
    mask = jr.bernoulli(key, keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def _call_base(base, x, *args, key, inference, **kwargs):
    call_kwargs = dict(kwargs)
    if key is not None:
        call_kwargs["key"] = key
    if inference is not None:
        call_kwargs["inference"] = inference
    try:
        return base(x, *args, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        return base(x, *args, **call_kwargs)


def _split_optional_key(key: jax.Array | None, count: int) -> tuple[jax.Array | None, ...]:
    if key is None:
        return (None,) * count
    return tuple(jr.split(key, count))


def _split_pair(key: jax.Array | None) -> tuple[jax.Array | None, jax.Array | None]:
    if key is None:
        return None, None
    key_a, key_b = jr.split(key, 2)
    return key_a, key_b


__all__ = (
    "AdaptFormerAdapter",
    "AdaptFormerBlock",
    "AdaptFormerConfig",
    "AdapterConfig",
    "BottleneckAdapter",
    "ParallelAdapterBlock",
    "SerialAdapterBlock",
    "add_adapter",
    "apply_adapters",
    "apply_adaptformer",
    "extract_adapter_delta",
    "iter_adapter_wrappers",
    "load_adapter_delta",
    "set_active_adapter",
    "strip_adapters",
)
