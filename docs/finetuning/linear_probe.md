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

For ViT-like backbones, `pool="cls_patch_mean"` concatenates the CLS token with
the mean over patch tokens. Prefix/register/distillation tokens are excluded
from the patch mean when model metadata is available, so the probe head input
width is `2 * dim`:

```python
probe = eqft.make_linear_probe(
    model,
    in_features=2 * model.dim,
    out_features=10,
    key=key,
    pool="cls_patch_mean",
)
```

If the transfer setup should train a fresh normalization layer for the
concatenated readout, wrap the probe head with `LayerNormReadoutHead`:

```python
head = eqft.LayerNormReadoutHead(
    2 * model.dim,
    eqft.LinearHead(2 * model.dim, 10, key=key),
)
probe = eqft.make_linear_probe(
    model,
    in_features=2 * model.dim,
    out_features=10,
    key=key,
    pool="cls_patch_mean",
    head=head,
)
```
