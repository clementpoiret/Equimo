# Prompts

Visual prompt tuning inserts trainable prompt tokens before ViT-like transformer
blocks. Prompt tokens are excluded from mean-patch pooling by default.

```python
prompted = eqft.apply_prompts(
    model,
    eqft.PromptConfig(num_tokens=10, depth="deep"),
    key=key,
)
plan = eqft.prepare_finetune(
    prompted,
    trainable=eqft.TrainableSpec(mode="peft", method_name="prompt"),
)
```

Prefix tuning wraps supported attention modules with trainable K/V prefixes.
Unsupported attention shapes raise a clear error. Prefix, scale/shift wrappers,
and IA3 wrappers use the same PEFT trainability path:

```python
model = eqft.apply_prefixes(model, key=key)
model = eqft.apply_scale_shift(model)
model = eqft.apply_ia3(model)
```
