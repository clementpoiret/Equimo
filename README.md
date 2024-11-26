# Equimo: Modern Vision Models in JAX/Equinox

**WARNING**: This is a research library implementing recent computer vision models. The implementations are based on paper descriptions and may not be exact replicas of the original implementations. Use with caution in production environments.

Equimo (Equinox Image Models) provides JAX/Equinox implementations of recent computer vision models, focusing on transformer and state-space architectures.

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
git clone https://github.com/yourusername/equimo.git
cd equimo
pip install -e .
```

## Implemented Models

| Model | Paper | Year | Status |
|-------|-------|------|--------|
| FasterViT | [FasterViT: Fast Vision Transformers with Hierarchical Attention](https://arxiv.org/abs/2306.06189) | 2023 | ✅ |
| MLLA | [Mamba-like Linear Attention](https://arxiv.org/abs/2405.16605) | 2024 | ✅ |
| PartialFormer | [PartialFormer: Going Beyond Learnable Pooling with Partial Attention](https://eccv.ecva.net/virtual/2024/poster/1877) | 2024 | ✅ |
| SHViT | [SHViT: Single Head Vision Transformer with Spatial-Channel Mixing](https://arxiv.org/abs/2401.16456) | 2024 | ✅ |
| VSSD | [VSSD: Vision Mamba with Non-Causal State Space Duality](https://arxiv.org/abs/2407.18559) | 2024 | ✅ |

## Basic Usage

```python
import jax
import equimo

# Create a model
model = equimo.models.FasterViT(
    img_size=224,
    in_channels=3,
    dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
)

# Generate random input
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (3, 224, 224))

# Run inference
output = model(x, enable_dropout=False, key=key)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Equimo in your research, please cite:

```bibtex
@software{equimo2024,
  author = {Your Name},
  title = {Equimo: Modern Vision Models in JAX/Equinox},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/equimo}
}
```
