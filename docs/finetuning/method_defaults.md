# Method Defaults

| Method | Default |
|---|---|
| Linear head | trunc-normal 0.02 weight init, zero bias |
| MLP head | 2 layers, GELU, hidden dim equals input dim |
| Linear probe | frozen backbone, trainable head |
| Full FT LLRD | decay 0.75, metadata only |
| LoRA | rank 8, alpha 16, zero-B identity init |
| rsLoRA | alpha over square-root rank scaling |
| DoRA | rank 8, alpha 16, base weight norm magnitude |
| Adapter | reduction factor 16, zero-up identity init |
| AdaptFormer | bottleneck 64, parallel branch, zero-up init |
| Prompt tuning | 10 tokens, deep, std 0.02 |
| Prefix tuning | 16 prefix tokens, deep metadata wrapper |
| Scale/shift | scale 1, shift 0 |
| IA3 | scaling vector 1 |
| WiSE-FT | alpha 0.5, head excluded by default |
