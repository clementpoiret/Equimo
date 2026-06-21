"""Side-tuning scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .._typing import PyTree
from ..heads import MLPHead


@dataclass(frozen=True)
class LSTConfig:
    """Ladder side-tuning metadata."""

    tap_layers: tuple[str, ...] = ("25%", "50%", "75%", "100%")
    side_width_multiplier: float = 0.25


class ActivationTap(eqx.Module):
    """Named activation tap pass-through module."""

    name: str = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x


class SideNetwork(eqx.Module):
    """Small side network over stopped backbone features."""

    head: MLPHead

    def __init__(self, in_features: int, out_features: int, *, key: jax.Array):
        self.head = MLPHead(in_features, out_features, key=key)

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
    side: SideNetwork
    ladder: LadderConnection

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        backbone_out = jax.lax.stop_gradient(self.backbone(x, **kwargs))
        side_out = self.side(backbone_out)
        return self.ladder(backbone_out, side_out)


__all__ = (
    "ActivationTap",
    "LSTConfig",
    "LadderConnection",
    "SideNetwork",
    "SideTunedModel",
)
