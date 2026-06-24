"""Minimal linear-probe example using a local Equinox model."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


class TinyBackbone(eqx.Module):
    dim: int = eqx.field(static=True)
    head: eqx.Module

    def __init__(self, *, key):
        self.dim = 4
        self.head = eqx.nn.Identity()

    def features(self, x):
        return jnp.stack([x, x + 1.0])


key = jr.PRNGKey(0)
model = TinyBackbone(key=key)
probe = eqft.make_linear_probe(
    model, in_features=4, out_features=3, key=key, pool="cls"
)
plan = eqft.prepare_finetune(probe, trainable=eqft.TrainableSpec(mode="head"))

print(plan.report)
