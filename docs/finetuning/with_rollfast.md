# Rollfast Integration

Rollfast remains external. Use Equimo plans as structural metadata and compile
the optimizer in Rollfast:

```python
import jax.numpy as jnp
import optax
import rollfast.finetune as rfft

optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    schedule="warmup_cosine",
    weight_decay=0.05,
    clip_global_norm=1.0,
    accumulation_steps=4,
    moment_dtype=jnp.float32,
)
opt_state = optim.init(plan.trainable)
```

Rollfast validates that `plan.trainable` and `plan.labels` have matching
PyTree structure, applies `plan.group_specs` learning-rate multipliers exactly
once, and keeps frozen leaves absent from optimizer state. Equimo core does not
import Rollfast.

For tag- or role-aware policies, prefer `group.tags_all`, `group.roles`, and
`group.mixed_roles`. The older `group.tags` and `group.role` fields are
representative metadata from the first leaf assigned to that optimizer label.

For memory-sensitive runs, Rollfast can use blockwise 8-bit AdamW state for
eligible optimizer moments. Equimo still emits the same plan; Rollfast decides
which groups are large and safe enough to quantize:

```python
optim = rfft.adamw8_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    state_quantization=rfft.StateQuantizationConfig(
        enabled=True,
        block_size=2048,
        min_size=4096,
    ),
)
```

The same plan can compile to Rollfast's structured optimizers:

```python
optim = rfft.hybrid_aurora_adam_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    weight_decay=0.05,
)
```

Use `hybrid_prism_adam_from_plan` for PRISM/Adam or
`hybrid_kron_adam_from_plan` for PSGD/Kron. Rollfast keeps the optimizer
partitioning and state; Equimo still owns the model-side plan and merge step.

SAM and ASAM also stay optimizer-side. Rollfast wraps a base optimizer with a
dedicated two-pass step and calls `plan.combine(...)` before each loss
evaluation:

```python
base = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    accumulation_steps=1,
)

step = rfft.make_sam_step(
    plan=plan,
    base_optimizer=base,
    config=rfft.SAMConfig(rho=0.05),
    loss_fn=loss_fn,
    microbatch_axis=0,
)
```

For ASAM, start with `ASAMConfig(rho=0.5, eta=0.01)` and sweep
the values for the task. Set `microbatch_axis` when the batch carries a leading
microbatch dimension; Rollfast accumulates both SAM gradient passes before the
single base optimizer update.

LoRA+ is an optimizer-side policy:

```python
optim = rfft.adamw_from_plan(
    lora_plan,
    total_steps=10_000,
    base_lr=2e-4,
    weight_decay=0.0,
    lora_b_lr_ratio=16.0,
)
```

AdaLoRA follows the same boundary. Equimo creates SVD-triplet AdaLoRA modules
and applies fixed-shape rank support masks; Rollfast owns the dynamic
rank-budget controller:

```python
rank_groups = eqft.lora_rank_groups(lora_model)
controller = rfft.make_adalora_controller(
    rank_groups,
    total_steps=20_000,
    config=rfft.AdaLoRAControllerConfig(initial_budget=12, target_budget=8),
)
state = controller.update(state, importance_scores, applied=True)
rank_pattern = controller.rank_pattern(state)
lora_model = eqft.apply_lora_rank_pattern(lora_model, rank_pattern)
```

`rank_pattern` keys use Equimo's canonical dot-separated LoRA module paths.

For staged workflows such as LP-FT, build a new Equimo plan and ask Rollfast to
migrate compatible optimizer state:

```python
stage2_bundle, stage2_state, migration = rfft.reconfigure_optimizer(
    old_plan=linear_probe_plan,
    old_bundle=stage1_bundle,
    old_state=stage1_state,
    new_plan=full_ft_plan,
    new_bundle=stage2_bundle,
    state_policy="preserve_shared",
    counter_policy="restart_schedule",
)
```

The migration report accounts for preserved, initialized, dropped, incompatible,
and group-changed leaves. With `state_policy="preserve_by_path_and_shape"`,
Rollfast can also preserve compatible Kron/PSGD preconditioner and Lipschitz
leaves by parameter path and factor shape. Equimo model parameters and deltas
remain separate.

Before initializing a Rollfast optimizer, estimate optimizer-family moment state
and Kron/PSGD preconditioner factors from the Equimo plan:

```python
estimate = rfft.estimate_optimizer_state_memory(
    plan,
    optim,
    preconditioner_dtype=jnp.bfloat16,
)
print(estimate.preconditioner_bytes)
print(estimate.warnings)
```

After initialization, inspect measured state memory from the actual optimizer
state:

```python
summary = rfft.optimizer_state_memory_summary(optim, opt_state)
print(summary.by_category)
print(summary.preconditioner_factors)
```

This is useful for Kron/PSGD and 8-bit AdamW, where actual state storage can
differ materially from a first-order AdamW estimate.

Schedule-Free Adam is also plan-aware. For validation and checkpointing, ask
Rollfast for the averaged evaluation parameters and combine them with the frozen
Equimo tree:

```python
optim = rfft.schedule_free_adam_from_plan(
    plan,
    total_steps=20_000,
    base_lr=5e-4,
    weight_decay=0.0,
)

updates, opt_state = optim.update(grads, opt_state, plan.trainable)
trainable = optax.apply_updates(plan.trainable, updates)
eval_model = plan.combine(optim.eval_params(trainable, opt_state))
```

EMA and SWA are optimizer-side evaluation views:

```python
optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    ema=rfft.EMAConfig(enabled=True, decay=0.9999),
    swa=rfft.SWAConfig(enabled=True, start_fraction=0.75),
)

eval_ema_model = plan.combine(
    optim.eval_params(trainable, opt_state, view="ema")
)
eval_swa_model = plan.combine(
    optim.eval_params(trainable, opt_state, view="swa")
)
```

Optimizer state is serialized by Rollfast, separately from Equimo model deltas:

```python
checkpoint = rfft.make_state_checkpoint(
    optim,
    opt_state,
    metadata={"step": 20_000},
)
opt_state = rfft.restore_state_checkpoint(optim, checkpoint)
```

For multi-device `pmap` training, pass the mapped axis name so global-norm
clipping reduces across devices:

```python
optim = rfft.adamw_from_plan(
    plan,
    total_steps=20_000,
    axis_name="data",
)
```
