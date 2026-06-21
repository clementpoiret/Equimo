# Model Merging

Equimo includes pure PyTree utilities:

```python
wise = eqft.interpolate_models(base, tuned, alpha=0.5, include_head=False)
soup = eqft.uniform_soup([model_a, model_b])
vector = eqft.task_vector(base, tuned)
reconstructed = eqft.apply_task_vector(base, vector)
```

Advanced primitives include TIES, DARE, Model Breadcrumbs, Fisher merge, and
RegMean. Fisher and RegMean require external statistics and fail clearly without
them.

PEFT wrappers that are algebraically mergeable also expose dedicated helpers:

```python
merged_lora = eqft.merge_lora(lora_model)
merged_dora = eqft.merge_dora(dora_model)
merged_ia3 = eqft.merge_ia3(ia3_model)
merged_scale_shift = eqft.merge_scale_shift(scale_shift_model)
merged_vera = eqft.merge_vera(vera_model)
```

Feature distillation accepts externally captured feature arrays directly, or
tap dictionaries/sequences selected by `FeatureDistillationConfig.layers`:

```python
loss = eqft.feature_distillation_loss_from_taps(
    student_taps,
    teacher_taps,
    config=eqft.FeatureDistillationConfig(layers=("50%", "100%")),
)

selected = eqft.select_feature_taps(student_taps, ("encoder.3", "100%"))
```
