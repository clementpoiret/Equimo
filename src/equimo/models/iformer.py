from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers import get_layer
from equimo.layers.generic import BlockChunk
from equimo.layers.norm import get_norm
from equimo.models.registry import register_model


@register_model("iformer")
class IFormer(eqx.Module):
    blocks: Tuple[BlockChunk, ...]
    dropout: eqx.nn.Dropout
    norm: type[eqx.Module]
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int,
        *,
        modules: list[str | type[eqx.Module] | None] = [None, None, None, None],
        module_kwargs: list[dict] = [{}, {}, {}, {}],
        downsamplers: list[str | type[eqx.Module] | None] = [None, None, None, None],
        downsampler_kwargs: list[dict] = [{}, {}, {}, {}],
        downsample_last: bool = False,
        dims: list[int] = [64, 128, 256, 512],
        depths: list[int] = [2, 2, 6, 2],
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        act_layer: str | Callable = "gelu",
        norm_layer: str | type[eqx.Module] = "layernorm",
        num_classes: int | None = 1000,
        eps=1e-5,
        init_values: float | None = None,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_down, key_blk, key_head = jr.split(key, 3)

        depth = sum(depths)

        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)
        modules = [get_layer(b) if b is not None else None for b in modules]
        downsamplers = [get_layer(b) if b is not None else None for b in downsamplers]

        universal_kwargs = {"act_layer": act_layer}

        if drop_path_uniform:
            # not traced
            dpr = [drop_path_rate] * depth
        else:
            # traced
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        blocks = []
        _bc_dim = [in_channels, *dims[:-1]]
        block_keys = jr.split(key_blk, len(dims))
        for i, _k in enumerate(block_keys):
            has_ds = downsamplers[i] is not None
            block_dim = _bc_dim[i] if (has_ds and downsample_last) else dims[i]
            mod_kw = universal_kwargs | module_kwargs[i]
            if modules[i] is not None:
                mod_kw = mod_kw | {"dim": block_dim}
            ds_kw = universal_kwargs | downsampler_kwargs[i]
            blocks.append(
                BlockChunk(
                    depth=depths[i],
                    in_channels=_bc_dim[i],
                    out_channels=dims[i],
                    module=modules[i],
                    module_kwargs=mod_kw,
                    downsampler=downsamplers[i],
                    downsampler_kwargs=ds_kw,
                    downsampler_needs_key=has_ds,
                    downsample_last=downsample_last,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    init_values=init_values,
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
    ) -> Float[Array, "num_classes"]:
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


def iformer_t(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=[
            "iformerblock",
            "iformerblock",
            "iformerblock",
            "shmablock",
            "iformerblock",
            "shmablock",
        ],
        module_kwargs=[
            {"kernel_size": 7, "expand_ratio": 3.0},
            {"kernel_size": 7, "expand_ratio": 3.0},
            {"kernel_size": 7, "expand_ratio": 3.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 2,
                "attn_ratio": 1.0,
                "ffn_ratio": 2.0,
            },
            {"kernel_size": 7, "expand_ratio": 3.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 4,
                "attn_ratio": 1.0,
                "ffn_ratio": 2.0,
            },
        ],
        downsamplers=[
            "iformerstem",
            "convnormdownsampler",
            "convnormdownsampler",
            None,
            None,
            "convnormdownsampler",
        ],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[32, 64, 128, 128, 128, 256],
        depths=[2, 2, 6, 3, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_s(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=[
            "iformerblock",
            "iformerblock",
            "iformerblock",
            "shmablock",
            "iformerblock",
            "shmablock",
        ],
        module_kwargs=[
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 2,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 4,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
        ],
        downsamplers=[
            "iformerstem",
            "convnormdownsampler",
            "convnormdownsampler",
            None,
            None,
            "convnormdownsampler",
        ],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[32, 64, 176, 176, 176, 320],
        depths=[2, 2, 9, 3, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_m(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=[
            "iformerblock",
            "iformerblock",
            "iformerblock",
            "shmablock",
            "iformerblock",
            "shmablock",
        ],
        module_kwargs=[
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 2,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 4,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
        ],
        downsamplers=[
            "iformerstem",
            "convnormdownsampler",
            "convnormdownsampler",
            None,
            None,
            "convnormdownsampler",
        ],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[48, 96, 192, 192, 192, 384],
        depths=[2, 2, 9, 4, 1, 2],
        **kwargs,
    )
    return backbone


# TODO: share windowing to avoid intermediate resizes
def iformer_m_faster(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=[
            "iformerblock",
            "iformerblock",
            "iformerblock",
            "shmablock",
            "iformerblock",
            "shmablock",
        ],
        module_kwargs=[
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 2,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
                "window_size": 16,
            },
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 4,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
        ],
        downsamplers=[
            "iformerstem",
            "convnormdownsampler",
            "convnormdownsampler",
            None,
            None,
            "convnormdownsampler",
        ],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[48, 96, 192, 192, 192, 384],
        depths=[2, 2, 9, 4, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_l(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=[
            "iformerblock",
            "iformerblock",
            "iformerblock",
            "shmablock",
            "iformerblock",
            "shmablock",
        ],
        module_kwargs=[
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 2,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 4,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
        ],
        downsamplers=[
            "iformerstem",
            "convnormdownsampler",
            "convnormdownsampler",
            None,
            None,
            "convnormdownsampler",
        ],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[48, 96, 256, 256, 256, 384],
        depths=[2, 2, 8, 8, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_l_faster(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=[
            "iformerblock",
            "iformerblock",
            "iformerblock",
            "shmablock",
            "iformerblock",
            "shmablock",
        ],
        module_kwargs=[
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 2,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
                "window_size": 16,
            },
            {"kernel_size": 7, "expand_ratio": 4.0},
            {
                "num_heads": 1,
                "head_dim_reduce_ratio": 4,
                "attn_ratio": 1.0,
                "ffn_ratio": 3.0,
            },
        ],
        downsamplers=[
            "iformerstem",
            "convnormdownsampler",
            "convnormdownsampler",
            None,
            None,
            "convnormdownsampler",
        ],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[48, 96, 256, 256, 256, 384],
        depths=[2, 2, 8, 8, 1, 2],
        **kwargs,
    )
    return backbone
