# Equimo: Modern Vision Models in JAX/Equinox

**WARNING**: This is a research library implementing recent computer vision models. The implementations are based on paper descriptions and may not be exact replicas of the original implementations. Use with caution in production environments.

Equimo (Equinox Image Models) provides JAX/Equinox implementations of recent computer vision models, currently focusing (but not limited to) on transformer and state-space architectures.

## Features

- Pure JAX/Equinox implementations
- Focus on recent architectures (2023-2024 papers)
- Modular design for easy experimentation
- Extensive documentation and type hints

## Installation

### From PyPI

```bash
pip install equimo
```

### From Source

```bash
git clone https://github.com/clementpoiret/equimo.git
cd equimo
pip install -e .
```

## Implemented Models

Beyond normal ViT (e.g., dinov2 or siglip), equimo proposes other SotA architectures:

| Model         | Paper                                                                                                                                                           | Year | Status    |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --------- |
| FasterViT     | [FasterViT: Fast Vision Transformers with Hierarchical Attention](https://arxiv.org/abs/2306.06189)                                                             | 2023 | ✅        |
| Castling-ViT  | [Castling-ViT: Compressing Self-Attention via Switching Towards Linear-Angular Attention During Vision Transformer Inference](https://arxiv.org/abs/2211.10526) | 2023 | Partial\* |
| MLLA          | [Mamba-like Linear Attention](https://arxiv.org/abs/2405.16605)                                                                                                 | 2024 | ✅        |
| PartialFormer | [Efficient Vision Transformers with Partial Attention](https://eccv.ecva.net/virtual/2024/poster/1877)                                                          | 2024 | ✅        |
| SHViT         | [SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design](https://arxiv.org/abs/2401.16456)                                                    | 2024 | ✅        |
| VSSD          | [VSSD: Vision Mamba with Non-Causal State Space Duality](https://arxiv.org/abs/2407.18559)                                                                      | 2024 | ✅        |

\*: Only contains the [Linear Angular Attention](https://github.com/clementpoiret/Equimo/blob/f8fcc79e45ca65e9deb1d970c4286c0b8562f9c2/equimo/layers/attention.py#L1407) module. It is straight forward to build a ViT around it, but may require an additional `__call__` kwarg to control the `sparse_reg` bool.

## Basic Usage

```python
import jax

import equimo.models as em

# Create a model (e.g. `faster_vit_0_224`)
key = jax.random.PRNGKey(0)
model = em.FasterViT(
    img_size=224,
    in_channels=3,
    dim=64,
    in_dim=64,
    depths=[2, 3, 6, 5],
    num_heads=[2, 4, 8, 16],
    hat=[False, False, True, False],
    window_size=[7, 7, 7, 7],
    ct_size=2,
    key=key,
)

# Generate random input
x = jax.random.normal(key, (3, 224, 224))

# Run inference
output = model(x, enable_dropout=False, key=key)
```

## Saving and Loading Models

Equimo provides utilities for saving models locally and loading pre-trained models from the
[official repository](https://huggingface.co/poiretclement/equimo).

### Saving Models Locally

```python
from pathlib import Path
from equimo.io import save_model

# Save model with compression (creates .tar.lz4 file)
save_model(
    Path("path/to/save/model"),
    model,  # can be any model you created using Equimo
    model_config,
    torch_hub_cfg,  # This can be an empty list, it's mainly to keep track of where are the weights coming
    compression=True
)

# Save model without compression (creates directory)
save_model(
    Path("path/to/save/model"),
    model,
    model_config,
    torch_hub_cfg,
    compression=False
)
```

### Loading Models

```python
from equimo.io import load_model

# Load a pre-trained model from the official repository
model = load_model(cls="vit", identifier="dinov2_vits14_reg")

# Load a local model (compressed)
model = load_model(cls="vit", path=Path("path/to/model.tar.lz4"))

# Load a local model (uncompressed directory)
model = load_model(cls="vit", path=Path("path/to/model/"))
```

Parameters passed to models can be overridden such as:

```python
model = load_model(
    cls="vit",
    identifier="siglip2_vitb16_256",
    dynamic_img_size=True,  # passed to the VisionTransformer class
)
```

#### List of pretrained models

Currently, only [DinoV2](https://github.com/facebookresearch/dinov2/) and SigLIP2 have been ported.

Model identifiers allows downloading from equimo's [repository on huggingface](https://huggingface.co/poiretclement/equimo/tree/main/models/default)

Identifiers are filenames without the extensions, such as:

- `dinov2_vitb14`
- `dinov2_vits14_reg`
- `siglip2_vitl16_512`
- `siglip2_vitso400m16_384`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Equimo in your research, please cite:

```bibtex
@software{equimo2024,
  author = {Clément POIRET},
  title = {Equimo: Modern Vision Models in JAX/Equinox},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/clementpoiret/equimo}
}
```
