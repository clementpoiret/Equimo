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
