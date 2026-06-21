"""Full fine-tuning plan with optional Optax integration."""

import equinox as eqx
import jax.random as jr

import equimo.finetune as eqft


model = eqx.nn.MLP(4, 2, 8, 2, key=jr.PRNGKey(0))
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="full"),
    labels=eqft.LLRDConfig(decay=1.0),
)

try:
    import optax
except ImportError:
    optax = None

if optax is not None:
    tx = optax.adamw(learning_rate=1e-4)
    opt_state = tx.init(plan.trainable)
    print(opt_state)
else:
    print(plan.report)
