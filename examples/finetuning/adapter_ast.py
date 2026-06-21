"""Adapter example using a local linear model as a stand-in."""

import equinox as eqx
import jax.random as jr

import equimo.finetune as eqft


model = eqx.nn.MLP(4, 2, 8, 2, key=jr.PRNGKey(0))
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="head"),
)

print(plan.report)
