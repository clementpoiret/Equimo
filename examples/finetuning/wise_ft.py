"""WiSE-FT interpolation example."""

import equinox as eqx
import jax.random as jr

import equimo.finetune as eqft


base = eqx.nn.MLP(4, 2, 8, 2, key=jr.PRNGKey(0))
tuned = eqx.nn.MLP(4, 2, 8, 2, key=jr.PRNGKey(1))
wise = eqft.interpolate_models(base, tuned, alpha=0.5, include_head=True)

print(type(wise).__name__)
