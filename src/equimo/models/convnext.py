__all__ = [
    "ConvNeXt",
    "convnext_sizes",
    "convnext_t",
    "convnext_s",
    "convnext_b",
    "convnext_l",
    "eupe_convnext_tiny",
    "eupe_convnext_small",
    "eupe_convnext_base",
]

from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.generic import BlockChunk
from equimo.layers.norm import get_norm
from equimo.models.registry import register_model

# Size configurations matching the original ConvNeXt paper.
convnext_sizes: dict[str, dict] = {
    "tiny": dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
    "small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    "base": dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
}


@register_model("convnext")
class ConvNeXt(eqx.Module):
    """ConvNeXt: A ConvNet for the 2020s (Liu et al., 2022).

    Four-stage hierarchical CNN using depthwise separable convolutions with
    inverted bottleneck design, LayerNorm2d, and GELU activations.

    Each stage consists of an optional downsampler followed by repeated
    ConvNeXtBlocks. The stem (stage 0) uses a stride-4 convolution for
    initial patchification; subsequent stages use stride-2 convolutions.

    Input convention: (C, H, W). Output: (num_classes,) or (dim,) features.
    """

    blocks: Tuple[BlockChunk, ...]
    dropout: eqx.nn.Dropout
    norm: eqx.Module
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int = 3,
        *,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        layer_scale_init_value: float = 1e-6,
        act_layer: str | Callable = "gelu",
        norm_layer: str | type[eqx.Module] = "layernorm",
        num_classes: int | None = 1000,
        eps: float = 1e-6,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_blk, key_head = jr.split(key, 2)

        depth = sum(depths)
        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        # Stage 0: ConvNeXtStem (4x downsample) + blocks
        # Stages 1-3: ConvNeXtDownsampler (2x downsample) + blocks
        downsamplers = [
            "convnextstem",
            "convnextdownsampler",
            "convnextdownsampler",
            "convnextdownsampler",
        ]
        _bc_dim = [in_channels, *dims[:-1]]

        blocks = []
        block_keys = jr.split(key_blk, len(dims))
        for i, _k in enumerate(block_keys):
            blocks.append(
                BlockChunk(
                    depth=depths[i],
                    in_channels=_bc_dim[i],
                    out_channels=dims[i],
                    module="convnextblock",
                    module_kwargs={
                        "dim": dims[i],
                        "act_layer": act_layer,
                    },
                    downsampler=downsamplers[i],
                    downsampler_kwargs={},
                    downsample_last=False,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    init_values=layer_scale_init_value,
                    key=_k,
                )
            )
        self.blocks = tuple(blocks)

        self.dropout = eqx.nn.Dropout(p=dropout)
        self.norm = norm_layer(dims[-1], eps=eps)
        self.head = (
            eqx.nn.Linear(dims[-1], num_classes, key=key_head)
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "dim height width"]:
        key_drop, *key_blocks = jr.split(key, len(self.blocks) + 1)

        for blk, key_blk in zip(self.blocks, key_blocks):
            x = blk(x, inference=inference, key=key_blk)
        x = self.dropout(x, inference=inference, key=key_drop)

        return x

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:
        x = self.features(x, inference=inference, key=key)
        x = self.norm(x.mean((1, 2)))
        x = self.head(x)

        return x


_CONVNEXT_BASE_CFG: dict = {
    "in_channels": 3,
    "layer_scale_init_value": 1e-6,
}

# EUPE ConvNeXt models are trained without a classification head (num_classes=0).
_EUPE_CONVNEXT_BASE_CFG: dict = {
    "in_channels": 3,
    "layer_scale_init_value": 1e-6,
    "num_classes": 0,
}

_CONVNEXT_REGISTRY: dict[str, tuple[dict, dict]] = {
    "convnext_t": (
        _CONVNEXT_BASE_CFG,
        convnext_sizes["tiny"],
    ),
    "convnext_s": (
        _CONVNEXT_BASE_CFG,
        convnext_sizes["small"],
    ),
    "convnext_b": (
        _CONVNEXT_BASE_CFG,
        convnext_sizes["base"],
    ),
    "convnext_l": (
        _CONVNEXT_BASE_CFG,
        convnext_sizes["large"],
    ),
    # EUPE pretrained variants
    "eupe_convnext_tiny": (
        _EUPE_CONVNEXT_BASE_CFG,
        {**convnext_sizes["tiny"], "act_layer": "exactgelu"},
    ),
    "eupe_convnext_small": (
        _EUPE_CONVNEXT_BASE_CFG,
        {**convnext_sizes["small"], "act_layer": "exactgelu"},
    ),
    "eupe_convnext_base": (
        _EUPE_CONVNEXT_BASE_CFG,
        {**convnext_sizes["base"], "act_layer": "exactgelu"},
    ),
}


def _build_convnext(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> ConvNeXt:
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _CONVNEXT_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    model = ConvNeXt(**cfg, key=key)

    if pretrained:
        from equimo.io import load_weights

        model = load_weights(
            model,
            identifier=variant,
            inference_mode=inference_mode,
        )

    return model


def convnext_t(**kwargs) -> ConvNeXt:
    """ConvNeXt-Tiny — dims [96,192,384,768], depths [3,3,9,3]."""
    return _build_convnext("convnext_t", **kwargs)


def convnext_s(**kwargs) -> ConvNeXt:
    """ConvNeXt-Small — dims [96,192,384,768], depths [3,3,27,3]."""
    return _build_convnext("convnext_s", **kwargs)


def convnext_b(**kwargs) -> ConvNeXt:
    """ConvNeXt-Base — dims [128,256,512,1024], depths [3,3,27,3]."""
    return _build_convnext("convnext_b", **kwargs)


def convnext_l(**kwargs) -> ConvNeXt:
    """ConvNeXt-Large — dims [192,384,768,1536], depths [3,3,27,3]."""
    return _build_convnext("convnext_l", **kwargs)


def eupe_convnext_tiny(pretrained: bool = False, **kwargs) -> ConvNeXt:
    """EUPE ConvNeXt-Tiny (LVD-1689M pretrained backbone, no classification head)."""
    return _build_convnext("eupe_convnext_tiny", pretrained=pretrained, **kwargs)


def eupe_convnext_small(pretrained: bool = False, **kwargs) -> ConvNeXt:
    """EUPE ConvNeXt-Small (LVD-1689M pretrained backbone, no classification head)."""
    return _build_convnext("eupe_convnext_small", pretrained=pretrained, **kwargs)


def eupe_convnext_base(pretrained: bool = False, **kwargs) -> ConvNeXt:
    """EUPE ConvNeXt-Base (LVD-1689M pretrained backbone, no classification head)."""
    return _build_convnext("eupe_convnext_base", pretrained=pretrained, **kwargs)
