from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.attention import RFAttentionBlock
from equimo.layers.convolution import DSConv, MBConv, SingleConvBlock
from equimo.layers.norm import get_norm


class BlockChunk(eqx.Module):
    residuals: list[bool] = eqx.field(static=True)
    blocks: list[DSConv | MBConv | RFAttentionBlock]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        *,
        key: PRNGKeyArray,
        block_type: Literal["conv", "attention"] = "conv",
        stride: int = 1,
        expand_ratio: float = 1.0,
        scales: Tuple[int, ...] = (5,),
        head_dim: int = 32,
        heads_ratio: float = 1.0,
        norm_layer: eqx.Module = eqx.nn.GroupNorm,
        act_layer: Callable = jax.nn.hard_swish,
        fewer_norm: bool = False,
        fuse_mbconv: bool = False,
        **kwargs,
    ):
        key, *block_subkeys = jr.split(key, depth + 1)

        keys_to_spread = [
            k for k, v in kwargs.items() if isinstance(v, list) and len(v) == depth
        ]

        blocks = []
        residuals = []

        # TODO: simplify logic
        match block_type:
            case "conv":
                block = DSConv if expand_ratio == 1.0 else MBConv
                if fewer_norm:
                    use_bias: Tuple[bool, ...] | bool = (
                        (True, False) if block == DSConv else (True, True, False)
                    )
                    norm_layer = (
                        (None, norm_layer)
                        if block == DSConv
                        else (None, None, norm_layer)
                    )
                else:
                    use_bias = False

                for i in range(depth):
                    config = kwargs | {k: kwargs[k][i] for k in keys_to_spread}

                    if block == MBConv:
                        config["expand_ratio"] = expand_ratio
                        config["fuse"] = fuse_mbconv

                    blocks.append(
                        block(
                            in_channels=in_channels if i == 0 else out_channels,
                            out_channels=out_channels,
                            stride=stride if i == 0 else 1,
                            use_bias=use_bias,
                            norm_layers=norm_layer,
                            act_layers=(act_layer, None)
                            if block == DSConv
                            else (act_layer, act_layer, None),
                            **config,
                            key=block_subkeys[i],
                        )
                    )
                    residuals.append(
                        (in_channels == out_channels and stride == 1) or i > 0
                    )

            case "attention":
                blocks.append(
                    MBConv(
                        in_channels,
                        out_channels,
                        stride=2,  # TODO: make downsampling optional
                        expand_ratio=expand_ratio,
                        norm_layers=(None, None, norm_layer),
                        act_layers=(act_layer, act_layer, None),
                        use_bias=(True, True, False),
                        fuse=fuse_mbconv,
                        key=key,
                    )
                )
                for i in range(depth):
                    blocks.append(
                        RFAttentionBlock(
                            in_channels=out_channels,
                            head_dim=head_dim,
                            heads_ratio=heads_ratio,
                            scales=scales,
                            rfattn_norm_layer=norm_layer,
                            expand_ratio=expand_ratio,
                            mbconv_norm_layers=(None, None, norm_layer),
                            mbconv_act_layers=(act_layer, act_layer, None),
                            fuse_mbconv=fuse_mbconv,
                            key=block_subkeys[i],
                        )
                    )
                residuals.append(False)

        self.blocks = blocks
        self.residuals = residuals

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "..."]:
        keys = jr.split(key, len(self.blocks))

        # TODO: Dropout and Stochastic Path Add
        for blk, residual, key_block in zip(self.blocks, self.residuals, keys):
            res = blk(x, inference=inference, key=key_block, **kwargs)
            x = x + res if residual else res

        return x


class ReduceFormer(eqx.Module):
    input_stem: eqx.nn.Sequential
    blocks: list[BlockChunk]
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int,
        widths: list[int],
        depths: list[int],
        block_types: list[Literal["conv", "attention"]],
        *,
        key: PRNGKeyArray,
        heads_dim: int = 32,
        expand_ratio: float = 4.0,
        norm_layer: eqx.Module | str = eqx.nn.GroupNorm,
        act_layer: Callable | str = jax.nn.hard_swish,
        fuse_mbconv: bool = False,
        num_classes: int | None = 1000,
        **kwargs,
    ):
        if not len(widths) == len(depths) == len(block_types):
            raise ValueError(
                "`widths`, `depths`, `strides`, and `expand_ratios` and `block_types` must have the same lengths."
            )

        key_stem, key_head, *key_blocks = jr.split(key, 3 + len(depths))

        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)

        width_stem = widths.pop(0)
        depth_stem = depths.pop(0)
        block_type_stem = block_types.pop(0)
        key_block_stem = key_blocks.pop(0)

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
                BlockChunk(
                    in_channels=width_stem,
                    out_channels=width_stem,
                    depth=depth_stem,
                    block_type=block_type_stem,
                    stride=1,
                    expand_ratio=1.0,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    key=key_block_stem,
                ),
            ]
        )

        self.blocks = [
            BlockChunk(
                in_channels=widths[i - 1] if i > 0 else width_stem,
                out_channels=widths[i],
                depth=depth,
                block_type=block_type,
                stride=2,
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                fuse_mbconv=fuse_mbconv,
                key=key_block,
            )
            for i, (depth, block_type, key_block) in enumerate(
                zip(depths, block_types, key_blocks)
            )
        ]

        self.head = (
            eqx.nn.Linear(
                in_features=widths[-1], out_features=num_classes, key=key_head
            )
            if num_classes and num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
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


def reduceformer_backbone_b1(**kwargs) -> ReduceFormer:
    backbone = ReduceFormer(
        widths=[16, 32, 64, 128, 256],
        depths=[1, 2, 3, 3, 4],
        block_types=[
            "conv",
            "conv",
            "conv",
            "attention",
            "attention",
        ],
        heads_dim=16,
        **kwargs,
    )
    return backbone


def reduceformer_backbone_b2(**kwargs) -> ReduceFormer:
    backbone = ReduceFormer(
        widths=[24, 48, 96, 192, 384],
        depths=[1, 3, 4, 4, 6],
        block_types=[
            "conv",
            "conv",
            "conv",
            "attention",
            "attention",
        ],
        heads_dim=32,
        **kwargs,
    )
    return backbone


def reduceformer_backbone_b3(**kwargs) -> ReduceFormer:
    backbone = ReduceFormer(
        widths=[32, 64, 128, 256, 512],
        depths=[1, 4, 6, 6, 9],
        block_types=[
            "conv",
            "conv",
            "conv",
            "attention",
            "attention",
        ],
        heads_dim=32,
        **kwargs,
    )
    return backbone
