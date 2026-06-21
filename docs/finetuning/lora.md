# LoRA

LoRA wraps selected linear modules with low-rank factors. Default initialization
uses random/kaiming `A` and zero `B`, so outputs match the base model.

```python
lora_model = eqft.apply_lora(
    model,
    eqft.LoRAConfig(rank=8, alpha=16.0),
    key=key,
)
plan = eqft.prepare_finetune(
    lora_model,
    trainable=eqft.TrainableSpec(mode="peft", method_name="lora"),
)
```

Merge for inference:

```python
merged = eqft.merge_lora(lora_model)
unmerged = eqft.unmerge_lora(merged)
```

Save and load a delta:

```python
eqft.save_delta(lora_model, "adapter.eqft", method="lora")
loaded = eqft.load_delta(base_model, "adapter.eqft")
```

rsLoRA, PiSSA initialization, DoRA, LoRA+ labels, and static rank masks are also
available through their config classes.
