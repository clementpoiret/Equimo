from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.attention import LowFormerBlock
from equimo.layers.convolution import DSConv, MBConv, SingleConvBlock
from equimo.layers.generic import BlockChunk
from equimo.layers.norm import get_norm
from equimo.models.registry import register_model


def _make_lowformer_chunk(
    block_type: Literal["conv", "attention"],
    in_channels: int,
    out_channels: int,
    depth: int,
    *,
    key: PRNGKeyArray,
    stride: int = 1,
    expand_ratio: float = 4.0,
    fuse_mbconv: bool = False,
    norm_layer,
    act_layer,
    drop_path: float | list[float] = 0.0,
    # attention-only params
    mlp_ratio: float = 4.0,
    att_stride: int = 1,
    attention_type: str = "softmax",
    attention_expand_ratio: float = 4.0,
) -> BlockChunk:
    is_ds = expand_ratio == 1.0
    if block_type == "conv":
        block_cls = DSConv if is_ds else MBConv
        mkw: dict = {
            "in_channels": [in_channels] + [out_channels] * max(depth - 1, 0),
            "out_channels": out_channels,
            "stride": [stride] + [1] * max(depth - 1, 0),
            "use_bias": False,
            "norm_layer": norm_layer,
            "act_layer": (act_layer, None) if is_ds else (act_layer, act_layer, None),
            "residual": True,
        }
        if not is_ds:
            mkw["expand_ratio"] = expand_ratio
            mkw["fuse"] = fuse_mbconv
        return BlockChunk(
            depth=depth,
            module=block_cls,
            module_kwargs=mkw,
            drop_path=drop_path,
            key=key,
        )
    else:  # attention
        return BlockChunk(
            depth=depth,
            in_channels=in_channels,
            out_channels=out_channels,
            module=LowFormerBlock,
            module_kwargs={
                "dim": out_channels,
                "mlp_ratio": mlp_ratio,
                "att_stride": att_stride,
                "attention_type": attention_type,
                "fuse_conv": True,
                "act_layer": act_layer,
                "norm_layer": norm_layer,
                "expand_ratio": expand_ratio,
                "mbconv_norm_layers": (None, None, norm_layer),
                "mbconv_act_layers": (act_layer, act_layer, None),
                "fuse_mbconv": fuse_mbconv,
            },
            downsampler=MBConv,
            downsampler_kwargs={
                "stride": 2,
                "expand_ratio": attention_expand_ratio,
                "norm_layer": (None, None, norm_layer),
                "act_layer": (act_layer, act_layer, None),
                "use_bias": (True, True, False),
                "fuse": fuse_mbconv,
            },
            downsampler_needs_key=True,
            downsample_last=False,
            drop_path=drop_path,
            key=key,
        )


@register_model("lowformer")
class LowFormer(eqx.Module):
    input_stem: eqx.nn.Sequential
    blocks: Tuple[eqx.Module, ...]
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int,
        widths: list[int],
        depths: list[int],
        att_strides: list[int],
        block_types: list[Literal["conv", "attention"]],
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 4.0,
        attention_type: Literal["softmax", "sigmoid"],
        stem_expand_ratio: float = 2.0,
        blocks_expand_ratio: float = 4.0,
        blocks_attention_expand_ratio: float = 4.0,
        norm_layer: str | type[eqx.Module] = "groupnorm",
        act_layer: str | Callable = "hard_swish",
        fuse_mbconv: bool = False,
        num_classes: int | None = 1000,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        **kwargs,
    ):
        if not len(widths) == len(depths) == len(att_strides) == len(block_types):
            raise ValueError(
                "`widths`, `depths`, `att_strides`, and `block_types` must have the same lengths."
            )

        key_stem, key_head, *key_blocks = jr.split(key, 3 + len(depths))

        depth = sum(depths)
        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)

        width_stem = widths.pop(0)
        depth_stem = depths.pop(0)
        block_type_stem = block_types.pop(0)
        key_block_stem = key_blocks.pop(0)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        self.input_stem = eqx.nn.Sequential(
            [
                SingleConvBlock(
                    in_channels=in_channels,
                    out_channels=width_stem,
                    kernel_size=3,
                    stride=2,
                    padding="SAME",
                    use_bias=False,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    key=key_stem,
                ),
                _make_lowformer_chunk(
                    block_type_stem,
                    width_stem,
                    width_stem,
                    depth_stem,
                    stride=1,
                    expand_ratio=stem_expand_ratio,
                    fuse_mbconv=fuse_mbconv,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    key=key_block_stem,
                ),
            ]
        )

        self.blocks = tuple(
            _make_lowformer_chunk(
                block_type,
                in_channels=widths[i - 1] if i > 0 else width_stem,
                out_channels=widths[i],
                depth=depth,
                stride=2,
                expand_ratio=blocks_expand_ratio,
                fuse_mbconv=fuse_mbconv,
                norm_layer=norm_layer,
                act_layer=act_layer,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                mlp_ratio=mlp_ratio,
                att_stride=att_stride,
                attention_type=attention_type,
                attention_expand_ratio=blocks_attention_expand_ratio,
                key=key_block,
            )
            for i, (depth, att_stride, block_type, key_block) in enumerate(
                zip(depths, att_strides, block_types, key_blocks)
            )
        )

        self.head = (
            eqx.nn.Linear(
                in_features=widths[-1], out_features=num_classes, key=key_head
            )
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "seqlen dim"]:
        """Extract features from input image.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Processed feature tensor
        """
        key_stem, *key_blocks = jr.split(key, len(self.blocks) + 1)

        x = self.input_stem(x, key=key_stem)

        for i, blk in enumerate(self.blocks):
            x = blk(x, inference=inference, key=key_blocks[i])

        return x

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:
        """Process input image through the full network.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Classification logits
        """
        x = self.features(x, inference=inference, key=key, **kwargs)

        x = reduce(x, "c h w -> c", "mean")

        x = self.head(x)

        return x


def lowformer_backbone_b0(**kwargs) -> LowFormer:
    backbone = LowFormer(
        widths=[8, 16, 32, 64, 128],
        depths=[0, 1, 1, 3, 4],
        block_types=[
            "conv",
            "conv",
            "conv",
            "attention",
            "attention",
        ],
        att_strides=[2, 2, 2, 2, 1],
        stem_expand_ratio=2.0,
        blocks_expand_ratio=4.0,
        blocks_attention_expand_ratio=4.0,
        fuse_mbconv=True,
        **kwargs,
    )
    return backbone


def lowformer_backbone_b1(**kwargs) -> LowFormer:
    backbone = LowFormer(
        widths=[16, 32, 64, 128, 256],
        depths=[1, 2, 3, 3, 4],
        block_types=[
            "conv",
            "conv",
            "conv",
            "attention",
            "attention",
        ],
        att_strides=[2, 2, 2, 2, 1],
        stem_expand_ratio=2.0,
        blocks_expand_ratio=4.0,
        blocks_attention_expand_ratio=4.0,
        fuse_mbconv=True,
        **kwargs,
    )
    return backbone


def lowformer_backbone_b2(**kwargs) -> LowFormer:
    backbone = LowFormer(
        widths=[24, 48, 96, 192, 384],
        depths=[1, 3, 4, 4, 6],
        block_types=[
            "conv",
            "conv",
            "conv",
            "attention",
            "attention",
        ],
        att_strides=[2, 2, 2, 2, 1],
        stem_expand_ratio=4.0,
        blocks_expand_ratio=4.0,
        blocks_attention_expand_ratio=6.0,
        fuse_mbconv=True,
        **kwargs,
    )
    return backbone


def lowformer_backbone_b3(**kwargs) -> LowFormer:
    backbone = LowFormer(
        widths=[32, 64, 128, 256, 512],
        depths=[1, 4, 6, 6, 9],
        block_types=[
            "conv",
            "conv",
            "conv",
            "attention",
            "attention",
        ],
        att_strides=[2, 2, 2, 2, 1],
        stem_expand_ratio=4.0,
        blocks_expand_ratio=6.0,
        blocks_attention_expand_ratio=6.0,
        fuse_mbconv=True,
        **kwargs,
    )
    return backbone
