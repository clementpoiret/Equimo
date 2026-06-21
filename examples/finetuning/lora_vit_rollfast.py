"""LoRA plan with optional Rollfast integration."""

import equinox as eqx
import jax.random as jr

import equimo.finetune as eqft


model = eqx.nn.MLP(4, 2, 8, 2, key=jr.PRNGKey(0))
lora_model = eqft.apply_lora(
    model,
    eqft.LoRAConfig(target=eqft.TargetSpec(predicate=eqft.is_linear), rank=2),
    key=jr.PRNGKey(1),
)
plan = eqft.prepare_finetune(
    lora_model,
    trainable=eqft.TrainableSpec(mode="peft", method_name="lora"),
)

try:
    import rollfast
except ImportError:
    rollfast = None

if rollfast is None:
    print(plan.report)
else:
    print("Build a Rollfast optimizer from plan.trainable, plan.labels, and plan.group_specs.")
