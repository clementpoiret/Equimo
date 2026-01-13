from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.attention import SHMABlock
from equimo.layers.convolution import (
    IFormerBlock,
    IFormerStem,
)
from equimo.layers.downsample import ConvNormDownsampler
from equimo.layers.norm import get_norm


def get_module(name, not_none: bool = True):
    match name:
        case "cndown":
            return ConvNormDownsampler
        case "ifb":
            return IFormerBlock
        case "ifs":
            return IFormerStem
        case "shma":
            return SHMABlock
        case None:
            if not_none and name is None:
                raise ValueError(f"{name} can't be None.")
            return None
        case _:
            raise NotImplementedError(f"{name} not implemented.")


@eqx.filter_jit
class BlockChunk(eqx.Module):
    downsample_last: bool = eqx.field(static=True)

    blocks: Tuple[eqx.Module, ...] | None
    downsample: eqx.Module | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        *,
        module: type[eqx.Module] | None = None,
        module_kwargs: dict = {},
        downsampler: type[eqx.Module] | None = None,
        downsampler_kwargs: dict = {},
        downsample_last: bool = False,
        drop_path: float | Sequence[float] = 0.0,
        init_values: float | None = None,
        key: PRNGKeyArray,
    ):
        assert module is not None or downsampler is not None, (
            "Either `module` or `downsampler` must be defined."
        )

        key_ds, *block_subkeys = jr.split(key, depth + 2)
        keys_to_spread = [
            k
            for k, v in module_kwargs.items()
            if isinstance(v, list) and len(v) == depth
        ]

        self.downsample_last = downsample_last

        # Handle channels for main modules and downsampler
        down_in = in_channels
        down_out = out_channels
        if downsampler is not None:
            block_in = block_out = in_channels if downsample_last else out_channels
        else:
            if depth > 1 and out_channels != in_channels:
                raise ValueError(
                    "Please use a dedicated downsampler to have a block of depth>1."
                )

            block_in = in_channels
            block_out = out_channels

        if module is not None:
            blocks = []
            for i in range(depth):
                config = module_kwargs | {
                    k: module_kwargs[k][i] for k in keys_to_spread
                }

                in_channels if downsample_last else out_channels
                blocks.append(
                    module(
                        in_channels=block_in,
                        out_channels=block_out,
                        drop_path=drop_path[i],
                        init_values=init_values,
                        **config,
                        key=block_subkeys[i],
                    )
                )
            self.blocks = tuple(blocks)
        else:
            # Only downsampler
            self.blocks = None

        self.downsample = (
            downsampler(
                in_channels=down_in,
                out_channels=down_out,
                **downsampler_kwargs,
                key=key_ds,
            )
            if downsampler is not None
            else None
        )

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        key: PRNGKeyArray,
        inference: bool = False,
        **kwargs,
    ) -> Tuple[Float[Array, "..."], list]:
        key_down, *keys = jr.split(key, len(self.blocks) + 1)

        if not self.downsample_last and self.downsample:
            x = self.downsample(x, inference=inference, key=key_down)

        for blk, key_block in zip(self.blocks, keys):
            x = blk(x, inference=inference, key=key_block, **kwargs)

        if self.downsample_last and self.downsample:
            x = self.downsample(x, inference=inference, key=key_down)

        return x


@eqx.filter_jit
class IFormer(eqx.Module):
    blocks: Tuple[BlockChunk, ...]
    dropout: eqx.nn.Dropout
    norm: type[eqx.Module]
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int,
        *,
        modules: list[str | None] = [None, None, None, None],
        module_kwargs: list[dict] = [{}, {}, {}, {}],
        downsamplers: list[str | None] = [None, None, None, None],
        downsampler_kwargs: list[dict] = [{}, {}, {}, {}],
        downsample_last: bool = False,
        dims: list[int] = [64, 128, 256, 512],
        depths: list[int] = [2, 2, 6, 2],
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        act_layer: Callable | str = jax.nn.gelu,
        norm_layer: str | type[eqx.Module] = eqx.nn.LayerNorm,
        num_classes: int = 1000,
        eps=1e-5,
        init_values: float | None = None,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_down, key_blk, key_head = jr.split(key, 3)

        depth = sum(depths)

        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)
        modules = [get_module(b, False) for b in modules]
        downsamplers = [get_module(b, False) for b in downsamplers]

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
            blocks.append(
                BlockChunk(
                    in_channels=_bc_dim[i],
                    out_channels=dims[i],
                    depth=depths[i],
                    module=modules[i],
                    module_kwargs=universal_kwargs | module_kwargs[i],
                    downsampler=downsamplers[i],
                    downsampler_kwargs=universal_kwargs | downsampler_kwargs[i],
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
            if num_classes > 0
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
        modules=["ifb", "ifb", "ifb", "shma", "ifb", "shma"],
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
        downsamplers=["ifs", "cndown", "cndown", None, None, "cndown"],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[32, 64, 128, 128, 128, 256],
        depths=[2, 2, 6, 3, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_s(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=["ifb", "ifb", "ifb", "shma", "ifb", "shma"],
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
        downsamplers=["ifs", "cndown", "cndown", None, None, "cndown"],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[32, 64, 176, 176, 176, 320],
        depths=[2, 2, 9, 3, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_m(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=["ifb", "ifb", "ifb", "shma", "ifb", "shma"],
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
        downsamplers=["ifs", "cndown", "cndown", None, None, "cndown"],
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
        modules=["ifb", "ifb", "ifb", "shma", "ifb", "shma"],
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
        downsamplers=["ifs", "cndown", "cndown", None, None, "cndown"],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[48, 96, 192, 192, 192, 384],
        depths=[2, 2, 9, 4, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_l(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=["ifb", "ifb", "ifb", "shma", "ifb", "shma"],
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
        downsamplers=["ifs", "cndown", "cndown", None, None, "cndown"],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[48, 96, 256, 256, 256, 384],
        depths=[2, 2, 8, 8, 1, 2],
        **kwargs,
    )
    return backbone


def iformer_l_faster(**kwargs) -> IFormer:
    backbone = IFormer(
        modules=["ifb", "ifb", "ifb", "shma", "ifb", "shma"],
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
        downsamplers=["ifs", "cndown", "cndown", None, None, "cndown"],
        downsampler_kwargs=[{}, {}, {}, {}, {}, {}],
        downsample_last=False,
        dims=[48, 96, 256, 256, 256, 384],
        depths=[2, 2, 8, 8, 1, 2],
        **kwargs,
    )
    return backbone
