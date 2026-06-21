# Serialization

`save_delta` writes a `FineTuneBundle` containing method metadata,
architecture hash, schema version, target paths, shapes, and method state.
The `.eqft` file is a pickle-free LZ4-compressed archive containing JSON
metadata and Equinox-serialized array leaves.

```python
bundle = eqft.save_delta(model, "delta.eqft", method="lora")
loaded = eqft.load_delta(base_model, "delta.eqft")
```

The spec-style order is also accepted:

```python
bundle = eqft.save_delta("delta.eqft", model, base_model, spec)
loaded = eqft.load_delta("delta.eqft", base_model)
```

Loading checks architecture hashes and target shapes. Incompatible bases raise
a method-specific error.

Use `save_finetune_bundle` and `load_finetune_bundle` when you already have a
`FineTuneBundle`. Bundle metadata includes parameter counts, dtype summary,
target paths, mergeability, and optional user metadata.

Supported delta methods: `lora`, `dora`, `adapter`, `prompt`, `prefix`,
`scale_shift`, and `ia3`.
