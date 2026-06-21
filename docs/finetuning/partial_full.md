# Full and Partial Fine-Tuning

Full fine-tuning selects all parameter leaves unless a freeze selector removes
them:

```python
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(
        mode="full",
        freeze=eqft.TargetSpec(tags_any=("embedding.patch",)),
    ),
)
```

Partial fine-tuning selects a depth range. Ranges are half-open:

```python
eqft.TrainableSpec(mode="partial", depth_range=(8, 12))
```
