# Linear Probe

Linear probing freezes the backbone and trains only a new head.

```python
probe = eqft.make_linear_probe(
    model,
    in_features=384,
    out_features=10,
    key=key,
    pool="cls",
)
plan = eqft.prepare_finetune(
    probe,
    trainable=eqft.TrainableSpec(mode="head"),
)
```

The wrapper replaces the original backbone head with an identity head so
head-only training selects only the probe head.
