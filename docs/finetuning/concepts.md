# Concepts

Equimo fine-tuning is model surgery plus PyTree planning.

Equinox does not use PyTorch-style `requires_grad`. A leaf is trainable when it
is present in the optimizer parameter PyTree:

```python
plan = eqft.prepare_finetune(model, trainable=eqft.TrainableSpec(mode="head"))
trainable = plan.trainable
frozen = plan.frozen
model = plan.combine(trainable)
```

Frozen leaves are absent from `plan.trainable`. For example, frozen patch
embedding parameters are not assigned a zero learning rate; they are removed
from optimizer state entirely.

`FineTunePlan` contains:

- `trainable`: parameters passed to the optimizer.
- `frozen`: parameters kept out of the optimizer.
- `labels`: optimizer-group labels matching trainable leaves.
- `group_specs`: learning-rate multiplier and weight-decay metadata.
- `param_info`: per-leaf path, tags, labels, and trainability.
- `report`: parameter counts and selected target paths.

Non-goals: optimizers, schedules, distributed training, dataloaders, feature
caches, experiment tracking, and training loops.
