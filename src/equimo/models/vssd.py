from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.attention import Attention, MllaBlock
from equimo.layers.convolution import Stem
from equimo.layers.ffn import Mlp
from equimo.layers.mamba import Mamba2Mixer
from equimo.layers.patch import PatchMerging
from equimo.models.vit import BlockChunk
from equimo.utils import to_list


class Vssd(eqx.Module):
    """Vision Mamba with Non-Causal State Space Duality (VSSD)[1].

    A hybrid vision architecture that combines Mamba state space models with attention
    mechanisms in a hierarchical structure. Features progressive spatial reduction
    and channel expansion through multiple stages.

    The model processes images through:
    1. Patch embedding using a stem layer
    2. Multiple stages of Mamba/Attention blocks with downsampling
    3. Global pooling and classification head

    Attributes:
        num_features: Number of features in final stage
        patch_embed: Initial patch embedding stem
        pos_drop: Positional dropout layer
        blocks: List of processing blocks with downsampling
        head: Classification head with normalization

    References:
        [1]: Shi, et al., 2024. https://arxiv.org/abs/2407.18559
    """

    num_features: int = eqx.field(static=True)

    patch_embed: eqx.Module
    pos_drop: eqx.Module
    blocks: List[eqx.Module]
    head: eqx.Module

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        repeat: int = 1,
        dim: int = 64,
        d_state: int = 64,
        d_conv: int = 3,
        expand: int = 2,
        patch_size: int = 4,
        depths: List[int] = [2, 4, 12, 4],
        num_heads: List[int] = [2, 4, 8, 16],
        attentions_layers: List[eqx.Module] | eqx.Module = [
            Mamba2Mixer,
            Mamba2Mixer,
            Mamba2Mixer,
            Attention,
        ],
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        mlp_ratio: float = 4.0,
        num_classes: int | None = 1000,
        **kwargs,
    ):
        """Initialize VSSD model.

        Args:
            img_size: Input image size
            in_channels: Number of input channels
            key: PRNG key for initialization
            repeat: Number of times to repeat each block
            dim: Initial model dimension
            d_state: Dimension of Mamba state space
            d_conv: Kernel size for Mamba convolution
            expand: Expansion factor for attention head dimension
            patch_size: Size of image patches
            depths: Number of blocks in each stage
            num_heads: Number of attention heads in each stage
            attentions_layers: Types of attention/mixing layers per stage
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth rate
            drop_path_uniform: Whether to use uniform drop path rate
            mlp_ratio: MLP expansion ratio
            num_classes: Number of classification classes
            **kwargs: Additional arguments
        """
        key_stem, key_head, *block_subkeys = jr.split(key, 2 + len(depths))

        n_chunks = len(depths)
        self.num_features = int(dim * 2 ** (n_chunks - 1))

        self.patch_embed = Stem(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=dim,
            key=key_stem,
        )
        patches_resolution = self.patch_embed.patches_resolution

        self.pos_drop = eqx.nn.Dropout(drop_rate)

        dpr = (
            list(jnp.linspace(0.0, drop_path_rate, n_chunks))
            if not drop_path_uniform
            else to_list(drop_path_rate, n_chunks)
        )

        num_heads = to_list(num_heads, n_chunks)
        attentions_layers = to_list(attentions_layers, n_chunks)
        self.blocks = [
            BlockChunk(
                block=MllaBlock,
                repeat=repeat,
                depth=depth,
                downsampler=PatchMerging if (i < n_chunks - 1) else eqx.nn.Identity,
                downsampler_contains_dropout=False,
                dim=int(dim * 2**i),
                input_resolution=(
                    patches_resolution[0] // (2**i),
                    patches_resolution[1] // (2**i),
                ),
                num_heads=num_heads[i],
                head_dim=int(dim * 2**i) * expand // num_heads[i],
                act_layer=jax.nn.gelu,
                use_dwc=False,
                attention_layer=attentions_layers[i],
                drop_path=dpr[i],
                mlp_ratio=mlp_ratio,
                ffn_layer=Mlp,
                key=block_subkeys[i],
            )
            for i, depth in enumerate(depths)
        ]

        self.norm = eqx.nn.LayerNorm(self.num_features)
        self.head = (
            eqx.nn.Linear(self.num_features, num_classes, key=key_head)
            if num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        key_pd, *keys = jr.split(key, 1 + len(self.blocks))

        x = self.patch_embed(x)
        x = self.pos_drop(x, inference=inference, key=key_pd)
        for i, blk in enumerate(self.blocks):
            x = blk(x, inference=inference, key=keys[i])

        return x

    def __call__(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        """Process input image through the VSSD network.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Classification logits for each class

        The forward pass consists of:
        1. Patch embedding and optional dropout
        2. Processing through multiple stages of blocks
        3. Global average pooling
        4. Classification head
        """
        x = self.features(x, inference=inference, key=key)
        x = jax.vmap(self.norm)(x)
        x = reduce(x, "s d -> d", "mean")
        x = self.head(x)

        return x
