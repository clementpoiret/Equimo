from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.attention import SHSA
from equimo.layers.convolution import ConvBlock, SingleConvBlock
from equimo.layers.generic import Residual
from equimo.layers.patch import SEPatchMerging
from equimo.models.vit import BlockChunk
from equimo.utils import nearest_power_of_2_divisor, to_list


class Downsampler(eqx.Module):
    """Downsampling module for spatial feature reduction.

    Combines convolution blocks and patch merging to reduce spatial dimensions
    while increasing feature channels. Uses residual connections and alternates
    between depthwise and pointwise convolutions.

    Attributes:
        conv1: First depthwise convolution with residual
        conv2: First pointwise convolution block
        conv3: Second depthwise convolution with residual
        conv4: Second pointwise convolution block
        patch_merging: Squeeze-and-excitation patch merging layer
    """

    conv1: eqx.Module
    conv2: eqx.Module
    conv3: eqx.Module
    conv4: eqx.Module
    patch_merging: eqx.Module

    def __init__(
        self,
        dim: int,
        out_dim: int,
        *,
        key: PRNGKeyArray,
        drop_path: float = 0.0,
        **kwargs,
    ):
        key_conv1, key_conv2, key_conv3, key_conv4, key_pm = jr.split(key, 5)
        self.conv1 = Residual(
            SingleConvBlock(
                dim,
                dim,
                act_layer=lambda x: x,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim,
                key=key_conv1,
            ),
            drop_path=drop_path,
        )
        self.conv2 = ConvBlock(
            dim,
            hidden_dim=dim * 2,
            act_layer=lambda x: x,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv2,
        )
        self.patch_merging = SEPatchMerging(
            dim=dim,
            out_dim=out_dim,
            key=key_pm,
        )
        self.conv3 = Residual(
            SingleConvBlock(
                out_dim,
                out_dim,
                act_layer=lambda x: x,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=out_dim,
                key=key_conv3,
            ),
            drop_path=drop_path,
        )
        self.conv4 = ConvBlock(
            out_dim,
            hidden_dim=out_dim * 2,
            act_layer=lambda x: x,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv4,
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        """Apply downsampling to input features.

        Args:
            x: Input feature tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Downsampled feature tensor with increased channels
        """
        key_conv1, key_conv2, key_conv3, key_conv4 = jr.split(key, 4)
        x = self.conv2(
            self.conv1(x, inference=inference, key=key_conv1),
            inference=inference,
            key=key_conv2,
        )
        x = self.patch_merging(x)
        x = self.conv4(
            self.conv3(x, inference=inference, key=key_conv3),
            inference=inference,
            key=key_conv4,
        )

        return x


class BasicBlock(eqx.Module):
    """Basic processing block combining convolution, attention and FFN.

    Processes features through three stages:
    1. Depthwise convolution with group normalization
    2. Optional SHSA (Single Head Self-Attention) mixer
    3. Feed-forward network using convolutions

    Attributes:
        conv: Depthwise convolution with normalization and residual
        mixer: SHSA attention block or identity if block_type != 's'
        ffn: Feed-forward network using convolutions
    """

    conv: Residual
    mixer: eqx.Module
    ffn: ConvBlock

    def __init__(
        self,
        dim: int,
        qk_dim: int,
        pdim: int,
        block_type: str,
        *,
        key: PRNGKeyArray,
        drop_path: float = 0.0,
        norm_max_group: int = 32,
        **kwargs,
    ):
        key_conv1, key_shsa, key_conv2 = jr.split(key, 3)
        num_groups = nearest_power_of_2_divisor(dim, norm_max_group)
        self.conv = Residual(
            eqx.nn.Sequential(
                [
                    eqx.nn.Conv(
                        num_spatial_dims=2,
                        in_channels=dim,
                        out_channels=dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=dim,
                        key=key_conv1,
                    ),
                    eqx.nn.GroupNorm(num_groups, dim),
                ]
            ),
            drop_path=drop_path,
        )
        self.mixer = (
            Residual(
                SHSA(
                    dim,
                    qk_dim,
                    pdim,
                    key=key_shsa,
                ),
                drop_path=drop_path,
            )
            if block_type == "s"
            else eqx.nn.Identity()
        )
        self.ffn = ConvBlock(
            dim,
            hidden_dim=int(dim**2),
            act_layer=jax.nn.relu,
            drop_path=drop_path,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv2,
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key_conv, key_mixer, key_ffn = jr.split(key, 3)
        return self.ffn(
            self.mixer(
                self.conv(x, inference=inference, key=key_conv),
                inference=inference,
                key=key_mixer,
            ),
            inference=inference,
            key=key_ffn,
        )


# TODO: Fix LayerSharing
class SHViT(eqx.Module):
    """Single Head Vision Transformer (SHViT)[1].

    A vision transformer that processes images through a simplified architecture
    combining convolutions and self-attention mechanisms. Features progressive
    spatial reduction and channel expansion through multiple stages.

    Attributes:
        patch_embed: Initial patch embedding through conv layers
        blocks: List of processing blocks with downsampling
        head: Classification head with normalization

    References:
        [1]: Yun, et al., 2024. https://arxiv.org/abs/2401.16456
    """

    patch_embed: eqx.nn.Sequential
    blocks: List[eqx.Module]
    head: eqx.Module

    def __init__(
        self,
        in_channels=3,
        dim: List[int] = [128, 256, 384],
        pdim: List[int] = [32, 64, 96],
        qk_dim=[16, 16, 16],
        depths: List[int] = [1, 2, 3],
        block_type=["s", "s", "s"],
        *,
        key: PRNGKeyArray,
        repeat: int = 1,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        num_classes: int | None = 1000,
        **kwargs,
    ):
        key_conv1, key_conv2, key_conv3, key_conv4, key_head, *block_subkeys = jr.split(
            key, 5 + len(depths)
        )

        n_chunks = len(depths)
        dims = to_list(dim, n_chunks)

        self.patch_embed = eqx.nn.Sequential(
            [
                SingleConvBlock(
                    in_channels=in_channels,
                    out_channels=dims[0] // 8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act_layer=jax.nn.relu,
                    key=key_conv1,
                ),
                SingleConvBlock(
                    in_channels=dims[0] // 8,
                    out_channels=dims[0] // 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act_layer=jax.nn.relu,
                    key=key_conv2,
                ),
                SingleConvBlock(
                    in_channels=dims[0] // 4,
                    out_channels=dims[0] // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act_layer=jax.nn.relu,
                    key=key_conv3,
                ),
                SingleConvBlock(
                    in_channels=dims[0] // 2,
                    out_channels=dims[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act_layer=lambda x: x,
                    key=key_conv4,
                ),
            ]
        )

        dpr = (
            list(jnp.linspace(0.0, drop_path_rate, n_chunks))
            if not drop_path_uniform
            else to_list(drop_path_rate, n_chunks)
        )

        qk_dims = to_list(qk_dim, n_chunks)
        pdims = to_list(pdim, n_chunks)
        block_types = to_list(block_type, n_chunks)
        self.blocks = [
            BlockChunk(
                block=BasicBlock,
                repeat=repeat,
                depth=depth,
                downsampler=Downsampler if i < len(depths) - 1 else eqx.nn.Identity,
                downsampler_contains_dropout=i < len(depths) - 1,
                downsampler_kwargs=(
                    {"out_dim": dims[i + 1]} if i < len(depths) - 1 else {}
                ),
                dim=dims[i],
                qk_dim=qk_dims[i],
                pdim=pdims[i],
                block_type=block_types[i],
                drop_path=dpr[i],
                key=block_subkeys[i],
            )
            for i, depth in enumerate(depths)
        ]

        self.norm = eqx.nn.LayerNorm(dims[-1])
        self.head = (
            eqx.nn.Linear(dims[-1], num_classes, key=key_head)
            if num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        keys = jr.split(key, len(self.blocks))

        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, inference=inference, key=keys[i])

        return x

    def __call__(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        """Process input image through the full SHViT network.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Classification logits for each class
        """
        x = self.features(x, inference=inference, key=key)
        x = jax.vmap(self.norm)(x)
        x = reduce(x, "c h w -> c", "mean")
        x = self.head(x)

        return x
