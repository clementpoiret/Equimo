# Method Defaults

| Method | Default |
|---|---|
| Linear head | trunc-normal 0.02 weight init, zero bias |
| MLP head | 2 layers, GELU, hidden dim equals input dim |
| Linear probe | frozen backbone, trainable head |
| Full FT LLRD | decay 0.75, metadata only |
| LoRA | rank 8, alpha 16, zero-B identity init |
| rsLoRA | alpha over square-root rank scaling |
| PiSSA | rank 16, truncated SVD LoRA initialization |
| LoRA+ | `lora_A` and `lora_B` labels only |
| static AdaLoRA | rank mask metadata, external scheduling |
| DoRA | rank 8, alpha 16, base weight norm magnitude |
| VeRA | rank 256, frozen random bases, trainable scale vectors |
| BitFit | bias leaves plus head, mask only |
| Adapter | reduction factor 16, after-MLP placement, zero-up identity init |
| AdaptFormer | bottleneck 64, parallel branch, zero-up init |
| Parallel adapters | residual-sum side branch |
| AdapterFusion | attention fusion over named adapter banks |
| Prompt tuning | 10 tokens, deep, std 0.02 |
| Soft prompts | 20 text prompt tokens, shallow |
| Deep prompts | 10 prompt tokens per layer |
| Prefix tuning | 16 prefix tokens, deep, projected K/V prefix state |
| Scale/shift | scale 1, shift 0 |
| IA3 | scaling vector 1 |
| Ladder side-tuning | 25/50/75/100 percent taps, width multiplier 0.25 |
| Continued SSL | LoRA r8/a16 plus last-block unfreeze metadata |
| L2-SP | unscaled mean penalty by default |
| Feature distillation | MSE over 50/100 percent layers |
| Mixout | p 0.1 anchored to pretrained weights |
| EWC | diagonal Fisher penalty from supplied statistics |
| WiSE-FT | alpha 0.5, head excluded by default |
| TIES | density 0.20, disjoint mean merge |
| DARE | drop rate 0.90 with rescaling |
| Model breadcrumbs | drop bottom 5 percent and top 1 percent deltas |
| Fisher merge | diagonal Fisher-weighted averaging |
| RegMean | ridge 1e-5, external input covariances |
| SAM/ASAM | metadata only, optimizer remains external |
