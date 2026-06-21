# Layer-Wise Learning-Rate Decay

LLRD labels blocks by original depth. Multipliers live in `GroupSpec`, not in
the label string.

For `n` blocks:

```python
lr_mult(i) = decay ** (n - 1 - i)
```

```python
plan = eqft.prepare_finetune(
    model,
    trainable=eqft.TrainableSpec(
        mode="full",
        freeze=eqft.TargetSpec(tags=("embedding.patch",)),
    ),
    labels=eqft.LLRDConfig(decay=0.75),
)
```

Patch embedding freeze means `patch_embed` leaves are absent from
`plan.trainable`; they do not get a zero learning rate.

Bias, norm, position embedding, class token, and mask token leaves are
no-weight-decay by default.
