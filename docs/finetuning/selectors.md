# Selectors

Selectors resolve parameter leaves by semantic tags, glob paths, predicates, and
depth constraints.

```python
eqft.TargetSpec(tags_any=("attention.qkv", "attention.proj"))
eqft.TargetSpec(include=("*.blocks.*.attn.qkv",))
eqft.TargetSpec(exclude=("*.pos_embed", "*.cls_token"))
eqft.TargetSpec(predicate=eqft.is_linear)
eqft.TargetSpec(tags_any=("block",), min_depth=8, max_depth=11)
```

Common tags include `embedding.patch`, `embedding.position`,
`embedding.class_token`, `embedding.register_token`, `block`,
`attention.qkv`, `attention.proj`, `mlp.fc1`, `mlp.fc2`, `norm`, and
`head`.

Empty selectors raise by default. Use explicit tags and inspect the result:

```python
paths = eqft.resolve_target_paths(model, eqft.TargetSpec(tags_any=("attention.qkv",)))
```
