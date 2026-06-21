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


def make_optax_tx_from_plan(plan, base_schedule, weight_decay=0.05):
    transforms = {}
    for label, group in plan.group_specs.items():
        wd = weight_decay if group.weight_decay else 0.0
        mult = group.lr_multiplier
        transforms[label] = optax.adamw(
            learning_rate=lambda step, mult=mult: base_schedule(step) * mult,
            weight_decay=wd,
        )
    return optax.partition(transforms, plan.labels)


try:
    import optax
except ImportError:
    optax = None

if optax is not None:
    tx = make_optax_tx_from_plan(plan, lambda step: 1e-4)
    opt_state = tx.init(plan.trainable)
    print(opt_state)
else:
    print(plan.report)
