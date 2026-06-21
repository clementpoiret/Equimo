"""LoRA plan with optional Rollfast grouped AdamW integration."""

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
    import rollfast.finetune as rfft
except ImportError:
    rfft = None

if rfft is None:
    print(plan.report)
else:
    optim = rfft.adamw_from_plan(
        plan,
        total_steps=1_000,
        base_lr=2e-4,
        schedule="warmup_cosine",
        weight_decay=0.0,
        lora_b_lr_ratio=16.0,
    )
    opt_state = optim.init(plan.trainable)
    del opt_state
    print(optim.report.group_table())
