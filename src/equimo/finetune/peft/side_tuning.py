"""Side-tuning scaffolding."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .._typing import Path, PyTree
from ..heads import MLPHead
from ..paths import key_path_to_path


@dataclass(frozen=True)
class LSTConfig:
    """Ladder side-tuning metadata."""

    stop_gradient_backbone: bool = True
    tap_layers: tuple[str, ...] = ("25%", "50%", "75%", "100%")
    side_width_multiplier: float = 0.25
    side_depth: int | Literal["num_taps"] = "num_taps"
    fusion: Literal["gated_sum"] = "gated_sum"
    gate_init: float = 0.0
    train_head: bool = True


class ActivationTap(eqx.Module):
    """Named activation tap pass-through module."""

    name: str = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class SideNetwork(eqx.Module):
    """Small side network over stopped backbone features."""

    head: MLPHead

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        key: jax.Array,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        head: MLPHead | None = None,
    ):
        self.head = (
            MLPHead(
                in_features,
                out_features,
                key=key,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
            if head is None
            else head
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(x)


class LadderConnection(eqx.Module):
    """Blend backbone and side outputs with a trainable gate."""

    gate: jax.Array

    def __init__(self, *, gate_init: float = 0.0):
        self.gate = jnp.asarray(gate_init, dtype=jnp.float32)

    def __call__(self, backbone: jax.Array, side: jax.Array) -> jax.Array:
        return backbone + self.gate * side


class SideTunedModel(eqx.Module):
    """Frozen-backbone side-tuning wrapper."""

    backbone: PyTree
    side: PyTree
    ladder: LadderConnection
    config: LSTConfig = eqx.field(static=True)

    def __init__(
        self,
        backbone: PyTree,
        side: PyTree,
        ladder: LadderConnection,
        config: LSTConfig | None = None,
    ):
        self.backbone = backbone
        self.side = side
        self.ladder = ladder
        self.config = LSTConfig() if config is None else config

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        backbone_out, side_input = _backbone_output_and_side_input(
            self.backbone,
            x,
            self.config,
            kwargs,
        )
        if self.config.stop_gradient_backbone:
            backbone_out = jax.lax.stop_gradient(backbone_out)
            side_input = jax.lax.stop_gradient(side_input)
        side_out = self.side(side_input)
        return self.ladder(backbone_out, side_out)


def apply_side_tuning(
    backbone: PyTree,
    *,
    in_features: int,
    out_features: int | None = None,
    key: jax.Array,
    config: LSTConfig | None = None,
) -> SideTunedModel:
    """Wrap a backbone with a trainable side network and ladder gate."""

    config = LSTConfig() if config is None else config
    if config.fusion != "gated_sum":
        raise ValueError("Side tuning currently supports fusion='gated_sum'.")
    out_features = in_features if out_features is None else out_features
    hidden_dim = max(1, int(round(in_features * config.side_width_multiplier)))
    num_layers = (
        max(1, len(config.tap_layers))
        if config.side_depth == "num_taps"
        else int(config.side_depth)
    )
    if num_layers < 1:
        raise ValueError("side_depth must resolve to at least one layer.")
    return SideTunedModel(
        backbone,
        SideNetwork(
            in_features,
            out_features,
            key=key,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ),
        LadderConnection(gate_init=config.gate_init),
        config,
    )


def iter_side_tuned_models(model: PyTree) -> tuple[tuple[Path, SideTunedModel], ...]:
    """Return path/module pairs for side-tuned wrappers."""

    return tuple(
        (key_path_to_path(key_path), leaf)
        for key_path, leaf in jtu.tree_leaves_with_path(
            model,
            is_leaf=lambda x: isinstance(x, SideTunedModel),
        )
        if isinstance(leaf, SideTunedModel)
    )


def strip_side_tuning(model: PyTree) -> PyTree:
    """Replace side-tuned wrappers with their backbones."""

    stripped = model
    for path, wrapper in iter_side_tuned_models(stripped):
        stripped = eqx.tree_at(
            lambda tree, p=path: _get_path(tree, p),
            stripped,
            wrapper.backbone,
        )
    return stripped


def _get_path(tree: PyTree, path: Path):
    node = tree
    for part in path:
        node = node[part] if isinstance(part, int) else getattr(node, part)
    return node


_OUTPUT_TAP_METHODS = ("call_with_taps", "forward_with_taps")
_FEATURE_TAP_METHODS = ("features_with_taps", "forward_features_with_taps")


def _backbone_output_and_side_input(
    backbone: PyTree,
    x: jax.Array,
    config: LSTConfig,
    kwargs: dict,
) -> tuple[jax.Array, jax.Array]:
    output = None
    taps = None

    for method_name in _OUTPUT_TAP_METHODS:
        if not hasattr(backbone, method_name):
            continue
        output, taps = _parse_tap_result(
            _call_with_optional_key(getattr(backbone, method_name), x, kwargs)
        )
        break

    if output is None:
        output = backbone(x, **kwargs)

    if taps is None:
        for method_name in _FEATURE_TAP_METHODS:
            if not hasattr(backbone, method_name):
                continue
            _, taps = _parse_tap_result(
                _call_with_optional_key(getattr(backbone, method_name), x, kwargs)
            )
            break

    if taps is None:
        return output, output

    return output, _combine_taps(_select_taps(taps, config.tap_layers))


def _call_with_optional_key(fn, x: jax.Array, kwargs: dict):
    call_kwargs = dict(kwargs)
    try:
        return fn(x, **call_kwargs)
    except TypeError as error:
        if "unexpected keyword argument" not in str(error):
            raise
        call_kwargs.pop("inference", None)
        try:
            return fn(x, **call_kwargs)
        except TypeError as second_error:
            if "unexpected keyword argument" not in str(second_error):
                raise
            call_kwargs.pop("key", None)
            return fn(x, **call_kwargs)


def _parse_tap_result(result) -> tuple[jax.Array | None, object | None]:
    if isinstance(result, tuple) and len(result) == 2:
        return result

    if isinstance(result, Mapping):
        output = _first_mapping_value(
            result,
            ("output", "out", "logits", "prediction", "predictions"),
        )
        taps = _first_mapping_value(result, ("taps", "activations", "intermediates"))
        if taps is None:
            taps = {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "output",
                    "out",
                    "logits",
                    "prediction",
                    "predictions",
                }
            }
        return output, taps

    return None, result


def _first_mapping_value(mapping: Mapping, keys: tuple[str, ...]):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _select_taps(taps: object, tap_layers: tuple[str, ...]) -> tuple[jax.Array, ...]:
    if isinstance(taps, Mapping):
        return _select_mapping_taps(taps, tap_layers)
    if isinstance(taps, (tuple, list)):
        return _select_sequence_taps(tuple(taps), tap_layers)
    return (jnp.asarray(taps),)


def _select_mapping_taps(
    taps: Mapping,
    tap_layers: tuple[str, ...],
) -> tuple[jax.Array, ...]:
    if not taps:
        raise ValueError("Tap-aware backbone returned no activations.")

    items = tuple(taps.items())
    values = tuple(value for _, value in items)
    selected: list[jax.Array] = []
    for selector in tap_layers:
        if selector in taps:
            selected.append(jnp.asarray(taps[selector]))
            continue
        if selector.isdigit() and int(selector) in taps:
            selected.append(jnp.asarray(taps[int(selector)]))
            continue
        if _is_percentage(selector):
            selected.append(jnp.asarray(values[_percentage_index(selector, len(values))]))
            continue
        raise ValueError(f"Tap selector {selector!r} was not found in backbone taps.")

    if selected:
        return tuple(selected)
    return (jnp.asarray(values[-1]),)


def _select_sequence_taps(
    taps: tuple[object, ...],
    tap_layers: tuple[str, ...],
) -> tuple[jax.Array, ...]:
    if not taps:
        raise ValueError("Tap-aware backbone returned no activations.")

    selected: list[jax.Array] = []
    for selector in tap_layers:
        if _is_percentage(selector):
            selected.append(jnp.asarray(taps[_percentage_index(selector, len(taps))]))
            continue
        if selector.isdigit():
            index = int(selector)
            if not 0 <= index < len(taps):
                raise ValueError(
                    f"Tap index {index} is out of range for {len(taps)} activations."
                )
            selected.append(jnp.asarray(taps[index]))
            continue
        raise ValueError(
            f"Tap selector {selector!r} requires named taps, but the backbone returned a sequence."
        )

    if selected:
        return tuple(selected)
    return (jnp.asarray(taps[-1]),)


def _is_percentage(selector: str) -> bool:
    if not selector.endswith("%"):
        return False
    try:
        _ = float(selector[:-1])
    except ValueError:
        return False
    return True


def _percentage_index(selector: str, length: int) -> int:
    value = float(selector[:-1])
    if not 0.0 < value <= 100.0:
        raise ValueError("Percentage tap selectors must be in the range (0%, 100%].")
    return min(length - 1, max(0, math.ceil(value / 100.0 * length) - 1))


def _combine_taps(taps: tuple[jax.Array, ...]) -> jax.Array:
    vectors = tuple(_tap_vector(tap) for tap in taps)
    first_shape = vectors[0].shape
    for vector in vectors[1:]:
        if vector.shape != first_shape:
            raise ValueError("Selected tap activations must pool to the same shape.")
    return jnp.mean(jnp.stack(vectors), axis=0)


def _tap_vector(tap: jax.Array) -> jax.Array:
    tap = jnp.asarray(tap)
    if tap.ndim == 0:
        return tap.reshape(1)
    if tap.ndim == 1:
        return tap
    return jnp.mean(tap, axis=tuple(range(tap.ndim - 1)))


__all__ = (
    "ActivationTap",
    "LSTConfig",
    "LadderConnection",
    "SideNetwork",
    "SideTunedModel",
    "apply_side_tuning",
    "iter_side_tuned_models",
    "strip_side_tuning",
)
