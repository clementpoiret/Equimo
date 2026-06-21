# Rollfast Integration

Rollfast remains external. Use Equimo plans as metadata:

```python
tx = rollfast.adamw_by_group(
    params=plan.trainable,
    labels=plan.labels,
    groups=plan.group_specs,
    base_lr=5e-4,
    schedule="warmup_cosine",
    weight_decay=0.05,
)
```

Equimo core does not import Rollfast.
