# Debugging Checklist

| Symptom | Likely cause | First fixes |
|---|---|---|
| linear probe poor | preprocessing mismatch or wrong feature pool | verify resize, normalization, sample rate, CLS vs mean pooling |
| PEFT underperforms probe | wrong target modules, LR too low | target more projections, adapters after both attn/MLP |
| full FT overfits immediately | data too small, augmentation weak | LoRA/adapters, fewer blocks, stronger regularization |
| dense task transfer poor | no spatial/dense head | dense adapter or task-specific head |
| ASR unstable | sequences too long, LR too high, CTC instability | shorter crops, warmup, lower LR externally |
| audio tagging flatlines | imbalance or weak segment coverage | balanced sampling externally, fixed crop strategy |
| AMP/bf16 NaNs | numerical instability | lower LR externally, disable path, test FP32 |
| prompt tuning collapses | too few tokens or poor init | deep prompts, more tokens, switch to LoRA/adapters |

- Print `plan.report` and confirm trainable parameter counts.
- Check `plan.report.target_paths` before training.
- Confirm frozen patch embedding leaves are absent from `plan.trainable`.
- Verify labels are not `None` for trainable leaves.
- Check `plan.group_specs[label].weight_decay` for norm and bias leaves.
- For PEFT, test identity initialization before training.
- For LoRA, compare merged and unmerged outputs before exporting.
- For deltas, load into a fresh compatible base model before publishing.
