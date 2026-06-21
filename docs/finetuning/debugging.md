# Debugging Checklist

- Print `plan.report` and confirm trainable parameter counts.
- Check `plan.report.target_paths` before training.
- Confirm frozen patch embedding leaves are absent from `plan.trainable`.
- Verify labels are not `None` for trainable leaves.
- Check `plan.group_specs[label].weight_decay` for norm and bias leaves.
- For PEFT, test identity initialization before training.
- For LoRA, compare merged and unmerged outputs before exporting.
- For deltas, load into a fresh compatible base model before publishing.
