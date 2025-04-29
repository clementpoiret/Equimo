# Equimo: Modern Vision Models in JAX/Equinox

**WARNING**: This is a research library implementing recent computer vision models. The implementations are based on paper descriptions and may not be exact replicas of the original implementations. Use with caution in production environments.

Equimo (Equinox Image Models) provides JAX/Equinox implementations of recent computer vision models, currently focusing (but not limited to) on transformer and state-space architectures.

## Features

- Pure JAX/Equinox implementations
- Focus on recent architectures (2023-2024 papers)
- Modular design for easy experimentation
- Extensive documentation and type hints
- **Experimental** support for text embedding

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
| ReduceFormer  | [ReduceFormer: Attention with Tensor Reduction by Summation](https://arxiv.org/abs/2406.07488)                                                                  | 2024 | ✅        |
| LowFormer     | [LowFormer: Hardware Efficient Design for Convolutional Transformer Backbones](https://arxiv.org/abs/2409.03460)                                                | 2024 | ✅        |

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

## Working with text embeddings

**Warning: this is experimental, it can break or change at any time**

`equimo.experimental.text` has been added since v0.3.0. It allows working with both text and images. It is especially
useful for models like SigLIP or TIPS, although only TIPS is currently supported.

Currently, text tokenization relies on tensorflow_text, install equimo with the `text` group such as
`uv add equimo[text]`.

Here is a very simple example of a 0-shot classification based on the comparison between text and image embeddings:

```python
import jax
from einops import rearrange

from equimo.experimental.text import Tokenizer
from equimo.io import load_image, load_model
from equimo.utils import PCAVisualizer, normalize, plot_image_and_feature_map

# Random demo inputs
key = jax.random.PRNGKey(42)
image = load_image("./demo.jpg", size=448)
text = [
    "A baby discovering happiness",
    "A computer",
]

# Loading pretrained models
image_encoder = load_model("vit", "tips_vits14_hr")
text_encoder = load_model("experimental.textencoder", "tips_vits14_hr_text")

# Encoding text and image
ids, paddings = Tokenizer(identifier="sentencepiece_tips").tokenize(text, max_len=64)

text_embedding = normalize(
    jax.vmap(text_encoder, in_axes=(0, 0, None))(ids, paddings, key)
)
image_embedding = jax.vmap(image_encoder.norm)(image_encoder.features(image, key))
cls_token = normalize(image_embedding[0])
spatial_features = rearrange(
    image_embedding[2:], "(h w) d -> h w d", h=int(448 / 14), w=int(448 / 14)
)

# Getting probabilities based on Cosine Similarity
cos_sim = jax.nn.softmax(
    ((cls_token[None, :] @ text_embedding.T) / text_encoder.temperature), axis=-1
)

# Plot the results
label_idxs = jax.numpy.argmax(cos_sim, axis=-1)
cos_sim_max = jax.numpy.max(cos_sim, axis=-1)
label_predicted = text[label_idxs[0]]
similarity = cos_sim_max[0]
pca_obj = PCAVisualizer(spatial_features)
image_pca = pca_obj(spatial_features)

plot_image_and_feature_map(
    image.transpose(1, 2, 0),
    image_pca,
    "./out.png",
    "Input Image",
    f"{label_predicted}, prob: {similarity * 100:.2f}%",
)
```

Resulting in such a wonderful result:

![Output of TIPS 0-shot classification](./docs/assets/0-shot.png)

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

The following models have pretrained weights available in Equimo:

- [DinoV2](https://arxiv.org/abs/2304.07193),
- [SigLIP2](https://arxiv.org/abs/2502.14786),
- [TIPS](https://arxiv.org/abs/2410.16512).

Model identifiers allow downloading from equimo's [repository on huggingface](https://huggingface.co/poiretclement/equimo/tree/main/models/default)

Identifiers are filenames without the extensions, such as:

- `dinov2_vitb14`
- `dinov2_vits14_reg`
- `siglip2_vitl16_512`
- `siglip2_vitso400m16_384`
- `tips_vitg14_lr`

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
