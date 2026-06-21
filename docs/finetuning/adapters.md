# Adapters

Bottleneck adapters wrap selected block submodules and preserve outputs at
initialization because the up projection starts at zero.

```python
adapted = eqft.apply_adapters(
    model,
    eqft.AdapterConfig(bottleneck=64, placement="after_mlp"),
    key=key,
)
plan = eqft.prepare_finetune(
    adapted,
    trainable=eqft.TrainableSpec(mode="peft", method_name="adapter"),
)
```

Supported placements include `after_mlp`, `parallel`, and `both`.
`after_mlp` wraps the block MLP output, `both` wraps both attention and MLP
outputs, and `parallel` adds a residual branch from the full block input.
AdaptFormer-style wrappers are available with `eqft.apply_adaptformer`.

Named adapter banks support serial placements:

```python
model = eqft.add_adapter(model, name="dataset_a", config=config, key=key_a)
model = eqft.add_adapter(model, name="dataset_b", config=config, key=key_b)
model = eqft.set_active_adapter(model, "dataset_a")
```

Adapter banks currently support `after_mlp` and `both`; use `apply_adapters`
directly for parallel adapters.

AdapterFusion trains a lightweight attention fusion module over existing named
serial adapter sites:

```python
fused = eqft.apply_adapter_fusion(
    model,
    eqft.AdapterFusionConfig(fusion="attention"),
    key=key,
)
plan = eqft.prepare_finetune(
    fused,
    trainable=eqft.adapter_fusion_trainable_spec(),
)
```
