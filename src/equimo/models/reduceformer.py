__all__ = [
    "ReduceFormer",
    "reduceformer_backbone_b1",
    "reduceformer_backbone_b2",
    "reduceformer_backbone_b3",
]

import copy
from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.attention import RFAttentionBlock
from equimo.layers.convolution import DSConv, MBConv, SingleConvBlock
from equimo.layers.generic import BlockChunk
from equimo.layers.norm import get_norm
from equimo.models.registry import register_model


def _make_reduceformer_chunk(
    block_type: Literal["conv", "attention"],
    in_channels: int,
    out_channels: int,
    depth: int,
    *,
    key: PRNGKeyArray,
    stride: int = 1,
    expand_ratio: float = 1.0,
    fuse_mbconv: bool = False,
    norm_layer,
    act_layer,
    dropout: float = 0.0,
    drop_path: float | list[float] = 0.0,
    residual: bool = False,
    # attention-only params
    head_dim: int = 32,
    heads_ratio: float = 1.0,
    scales: Tuple[int, ...] = (5,),
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
            "dropout": dropout,
            "residual": residual,
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
            module=RFAttentionBlock,
            module_kwargs={
                "in_channels": out_channels,
                "head_dim": head_dim,
                "heads_ratio": heads_ratio,
                "scales": scales,
                "rfattn_norm_layer": norm_layer,
                "expand_ratio": expand_ratio,
                "mbconv_norm_layers": (None, None, norm_layer),
                "mbconv_act_layers": (act_layer, act_layer, None),
                "fuse_mbconv": fuse_mbconv,
                "context_drop": dropout,
                "local_drop": dropout,
                "residual_mbconv": False,
            },
            downsampler=MBConv,
            downsampler_kwargs={
                "stride": 2,
                "expand_ratio": expand_ratio,
                "norm_layer": (None, None, norm_layer),
                "act_layer": (act_layer, act_layer, None),
                "use_bias": (True, True, False),
                "dropout": dropout,
                "fuse": fuse_mbconv,
                "residual": False,
            },
            downsampler_needs_key=True,
            downsample_last=False,
            drop_path=drop_path,
            key=key,
        )


@register_model("reduceformer")
class ReduceFormer(eqx.Module):
    conv_stem: SingleConvBlock
    block_stem: eqx.Module
    blocks: Tuple[eqx.Module, ...]
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int,
        widths: list[int],
        depths: list[int],
        block_types: list[Literal["conv", "attention"]],
        *,
        key: PRNGKeyArray,
        head_dim: int = 32,
        expand_ratio: float = 4.0,
        norm_layer: str | type[eqx.Module] = "groupnorm",
        act_layer: str | Callable = "hard_swish",
        fuse_mbconv: bool = False,
        num_classes: int | None = 1000,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        residual: bool = False,
        **kwargs,
    ):
        if not len(widths) == len(depths) == len(block_types):
            raise ValueError(
                "`widths`, `depths`, `strides`, and `expand_ratios` and `block_types` must have the same lengths."
            )

        key_stem, key_head, *key_blocks = jr.split(key, 3 + len(depths))

        depth = sum(depths)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)

        width_stem = widths.pop(0)
        depth_stem = depths.pop(0)
        block_type_stem = block_types.pop(0)
        key_block_stem = key_blocks.pop(0)
        dpr_stem = 0.0

        self.conv_stem = SingleConvBlock(
            in_channels=in_channels,
            out_channels=width_stem,
            kernel_size=3,
            stride=2,
            padding="SAME",
            use_bias=False,
            norm_layer=norm_layer,
            act_layer=act_layer,
            key=key_stem,
        )
        self.block_stem = _make_reduceformer_chunk(
            block_type_stem,
            width_stem,
            width_stem,
            depth_stem,
            stride=1,
            expand_ratio=1.0,
            norm_layer=norm_layer,
            act_layer=act_layer,
            dropout=dropout,
            residual=residual,
            drop_path=dpr_stem,
            key=key_block_stem,
        )

        self.blocks = tuple(
            _make_reduceformer_chunk(
                block_type,
                in_channels=widths[i - 1] if i > 0 else width_stem,
                out_channels=widths[i],
                depth=depth,
                stride=2,
                expand_ratio=expand_ratio,
                fuse_mbconv=fuse_mbconv,
                norm_layer=norm_layer,
                act_layer=act_layer,
                dropout=dropout,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                residual=residual,
                head_dim=head_dim,
                key=key_block,
            )
            for i, (depth, block_type, key_block) in enumerate(
                zip(depths, block_types, key_blocks)
            )
        )

        self.head = (
            eqx.nn.Linear(
                in_features=widths[-1], out_features=num_classes, key=key_head
            )
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )

    def intermediates(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ):
        key_stem, *key_blocks = jr.split(key, len(self.blocks) + 1)

        intermediates = []

        x = self.conv_stem(x, inference=inference, key=key_stem)
        x = self.block_stem(x, inference=inference, key=key_stem)

        intermediates.append(x)  # let's consider the stem is a normal block

        for i, blk in enumerate(self.blocks):
            x = blk(x, inference=inference, key=key_blocks[i])
            intermediates.append(x)

        return intermediates

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

        x = self.conv_stem(x, inference=inference, key=key_stem)
        x = self.block_stem(x, inference=inference, key=key_stem)

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


_REDUCEFORMER_BASE_CFG: dict = {
    "in_channels": 3,
    "block_types": ["conv", "conv", "conv", "attention", "attention"],
}

_REDUCEFORMER_REGISTRY: dict[str, tuple[dict, dict]] = {
    "reduceformer_backbone_b1": (
        _REDUCEFORMER_BASE_CFG,
        {
            "widths": [16, 32, 64, 128, 256],
            "depths": [1, 2, 3, 3, 4],
            "head_dim": 16,
        },
    ),
    "reduceformer_backbone_b2": (
        _REDUCEFORMER_BASE_CFG,
        {
            "widths": [24, 48, 96, 192, 384],
            "depths": [1, 3, 4, 4, 6],
            "head_dim": 32,
        },
    ),
    "reduceformer_backbone_b3": (
        _REDUCEFORMER_BASE_CFG,
        {
            "widths": [32, 64, 128, 256, 512],
            "depths": [1, 4, 6, 6, 9],
            "head_dim": 32,
        },
    ),
}


def _build_reduceformer(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> ReduceFormer:
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _REDUCEFORMER_REGISTRY[variant]
    cfg = copy.deepcopy(base_cfg | variant_cfg | overrides)
    model = ReduceFormer(**cfg, key=key)

    if pretrained:
        from equimo.io import load_weights

        model = load_weights(
            model,
            identifier=variant,
            inference_mode=inference_mode,
        )

    return model


def reduceformer_backbone_b1(**kwargs) -> ReduceFormer:
    """ReduceFormer-B1 — widths [16→256], depths [1,2,3,3,4], head_dim 16."""
    return _build_reduceformer("reduceformer_backbone_b1", **kwargs)


def reduceformer_backbone_b2(**kwargs) -> ReduceFormer:
    """ReduceFormer-B2 — widths [24→384], depths [1,3,4,4,6], head_dim 32."""
    return _build_reduceformer("reduceformer_backbone_b2", **kwargs)


def reduceformer_backbone_b3(**kwargs) -> ReduceFormer:
    """ReduceFormer-B3 — widths [32→512], depths [1,4,6,6,9], head_dim 32."""
    return _build_reduceformer("reduceformer_backbone_b3", **kwargs)
