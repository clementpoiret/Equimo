# LP-FT

LP-FT is a two-stage workflow:

1. Train a linear probe or head-only model.
2. Continue from that trained head while unfreezing the backbone.

```python
recipe = eqft.lpft()
stage1_plan = recipe.stage1_plan(model)

# external training updates the head here

stage2_plan = recipe.stage2_plan(model_with_trained_head)
```

Equimo preserves the trained head because it only changes trainability and
labels. Optimizers and schedules are external.

`LPFTRecipe.stage2_plan()` currently supports the paper-style preserved-head
transition. To intentionally reset or replace the head, do that on the model
before building the stage-2 plan.
