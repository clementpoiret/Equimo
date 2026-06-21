# Equimo Fine-Tuning

`equimo.finetune` provides model-side adaptation primitives for Equinox
PyTrees: selectors, trainability masks, labels, reports, heads, PEFT wrappers,
deltas, and merge utilities.

It does not provide optimizers, schedules, dataloaders, trainers, or experiment
tracking. The usual flow is:

```python
import equimo.finetune as eqft

plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(mode="full"),
    labels=eqft.LLRDConfig(decay=0.75),
)
```

Use `plan.trainable`, `plan.frozen`, `plan.labels`, and `plan.group_specs` with
Optax, Rollfast, or your own training loop.

Start with these pages:

- [Concepts](concepts.md)
- [Selectors](selectors.md)
- [Linear probe](linear_probe.md)
- [LP-FT](lpft.md)
- [LLRD](llrd.md)
- [LoRA](lora.md)
- [Adapters](adapters.md)
- [Prompts](prompts.md)
- [Full and partial fine-tuning](partial_full.md)
- [Model merging](model_merging.md)
- [Serialization](serialization.md)
- [Optax integration](with_optax.md)
- [Rollfast integration](with_rollfast.md)
- [Debugging](debugging.md)
- [Method defaults](method_defaults.md)
- [Reference anchors](references.md)
