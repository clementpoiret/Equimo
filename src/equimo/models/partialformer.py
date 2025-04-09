from typing import Callable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange, reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.attention import PartialFormerBlock
from equimo.layers.convolution import Stem
from equimo.layers.ffn import Mlp
from equimo.layers.patch import PatchMerging
from equimo.layers.posemb import PosCNN
from equimo.layers.sharing import LayerSharing
from equimo.utils import to_list


class LayerSharingWithQA(LayerSharing):
    """Layer sharing implementation with Query Attention (QA) token support.

    Extends LayerSharing to handle query attention tokens while maintaining
    layer sharing functionality. Processes both input features and QA tokens
    through shared layers with optional LoRA adaptations.
    """

    def __call__(
        self,
        x: Array,
        qa: Array,
        *args,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ):
        if self.repeat == 1:
            return self.f(
                x,
                *args,
                inference=inference,
                key=key,
                **kwargs,
            )

        keys = jr.split(key, self.repeat)
        reshape = len(x.shape) == 3

        for i in range(self.repeat):
            if reshape:
                _, h, w = x.shape
                lora_x = rearrange(x, "c h w -> (h w) c")
            else:
                lora_x = x
            lora_output = self.dropouts[i](
                jax.vmap(self.loras[i])(lora_x),
                inference=inference,
                key=keys[i],
            )
            if reshape:
                lora_output = rearrange(
                    lora_output,
                    "(h w) c -> c h w",
                    h=h,
                    w=w,
                )

            x, qa = self.f(
                x,
                qa=qa,
                inference=inference,
                key=key,
                **kwargs,
            )

            x += lora_output

        return x, qa


class BlockChunk(eqx.Module):
    """A chunk of processing blocks with query attention and downsampling support.

    Processes input features through a sequence of attention blocks while
    maintaining and updating a query attention token. Includes positional
    embeddings, downsampling, and QA token projection capabilities.

    Attributes:
        reshape: Whether to reshape inputs for processing
        downsampler_contains_dropout: If downsampler has dropout
        posemb: Positional embedding layer
        blocks: List of processing blocks
        downsample: Downsampling layer
        qa_proj: Query attention projection layer
        qa_drop: Dropout for query attention
    """

    reshape: bool = eqx.field(static=True)
    downsampler_contains_dropout: bool = eqx.field(static=True)

    posemb: eqx.Module
    blocks: List[eqx.Module]
    downsample: eqx.Module
    qa_proj: eqx.Module
    qa_drop: eqx.Module

    def __init__(
        self,
        depth: int,
        *,
        key: PRNGKeyArray,
        block: eqx.Module = PartialFormerBlock,
        use_cpe: bool = False,
        qa_act_layer: Callable = jax.nn.relu,
        qa_norm_layer: eqx.Module = eqx.nn.LayerNorm,
        qa_drop: float = 0.0,
        repeat: int = 1,
        downsampler: eqx.Module = eqx.nn.Identity,
        downsampler_contains_dropout: bool = False,
        downsampler_kwargs: dict = {},
        **kwargs,
    ):
        key_ds, key_pos, key_qaproj, *block_subkeys = jr.split(key, depth + 3)
        if not isinstance(downsampler, eqx.nn.Identity) or use_cpe:
            if kwargs.get("dim") is None:
                raise ValueError(
                    "Using a downsampler or a CPE requires passing a `dim` argument."
                )

        # self.reshape = block is not ConvBlock
        self.reshape = True  # TODO
        self.downsampler_contains_dropout = downsampler_contains_dropout

        keys_to_spread = [
            k for k, v in kwargs.items() if isinstance(v, list) and len(v) == depth
        ]

        dim = kwargs.get("dim")
        self.posemb = (
            PosCNN(
                dim,
                dim,
                key=key_pos,
            )
            if use_cpe
            else eqx.nn.Identity()
        )

        blocks = []
        for i in range(depth):
            config = kwargs | {k: kwargs[k][i] for k in keys_to_spread}
            blocks.append(
                LayerSharingWithQA(
                    dim=dim,
                    f=block(**config, key=block_subkeys[i]),
                    repeat=repeat,
                    key=block_subkeys[i],
                ),
            )
        self.blocks = blocks

        self.downsample = downsampler(dim=dim, **downsampler_kwargs, key=key_ds)
        self.qa_proj = (
            eqx.nn.Sequential(
                [
                    eqx.nn.Linear(
                        dim,
                        dim * 2 if downsampler is not eqx.nn.Identity else dim,
                        key=key_qaproj,
                    ),
                    qa_norm_layer(dim * 2),
                    eqx.nn.Lambda(qa_act_layer),
                ]
            )
            if downsampler is not eqx.nn.Identity
            else eqx.nn.Identity()
        )
        self.qa_drop = eqx.nn.Dropout(qa_drop)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        qa: Float[Array, "1 dim"],
        *,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Process input features and query attention token.

        Args:
            x: Input feature tensor
            qa: Query attention token
            inference: Whether to enable dropout
            key: PRNG key for random operations

        Returns:
            Tuple of (processed features, updated query attention token)
        """
        key_qadrop, *keys = jr.split(key, len(self.blocks) + 1)

        x = self.posemb(x)

        for blk, key_block in zip(self.blocks, keys):
            x, qa = blk(x, qa=qa, inference=inference, key=key_block, **kwargs)

        if self.downsampler_contains_dropout:
            x = self.downsample(x, inference=inference, key=key)
        else:
            x = self.downsample(x)
        qa = self.qa_drop(
            jax.vmap(self.qa_proj)(qa),
            inference=inference,
            key=key_qadrop,
        )

        return x, qa


class PartialFormer(eqx.Module):
    """PartialFormer implementation with a Partial Attention mechanism[1].

    A vision transformer that processes images through patches while using
    a query attention token to guide feature extraction. Combines hierarchical
    feature processing with query-based attention mechanisms.

    Attributes:
        num_features: Number of features in final layer
        qa_token: Query attention token
        patch_embed: Patch embedding layer
        pos_drop: Positional dropout
        blocks: Processing blocks
        norm: Normalization layer
        head: Classification head

    Notes:
        WARNING, the original paper[1] does not provide an official
        implementation. This implementation if an interpretation of the paper.
        Although I made my best to follow what I read, it may not be a 1:1
        replica.

    References:
        [1]: Vo, et al., 2024. https://eccv.ecva.net/virtual/2024/poster/1877
    """

    num_features: int = eqx.field(static=True)

    qa_token: jnp.ndarray
    patch_embed: Stem
    pos_drop: eqx.nn.Dropout
    blocks: List[eqx.Module]
    norm: eqx.Module
    head: eqx.Module

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        dim: int,
        num_heads: int | List[int],
        depths: List[int],
        foreground_ratios: Tuple[float, float] | float,
        *,
        key: PRNGKeyArray,
        patch_size: int = 7,
        pos_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        block: eqx.Module = PartialFormerBlock,
        repeat: int = 1,
        head_expand_ratio: float = 4.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        proj_ratio: float = 4.0,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = jax.nn.gelu,
        ffn_layer: eqx.Module = Mlp,
        ffn_bias: bool = True,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        init_values: float | None = None,
        num_classes: int = 1000,
        interpolate_antialias: bool = False,
        **kwargs,
    ):
        depth = sum(depths)
        key_qa, key_stem, key_head, *block_subkeys = jr.split(key, 3 + len(depths))

        self.qa_token = jr.uniform(key_qa, (1, dim))

        self.patch_embed = Stem(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=dim,
            key=key_stem,
        )

        self.pos_drop = eqx.nn.Dropout(pos_drop_rate)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = list(jnp.linspace(0.0, drop_path_rate, depth))

        if isinstance(foreground_ratios, float):
            f_ratios = [foreground_ratios] * depth
        elif isinstance(foreground_ratios, Tuple[float, float]):
            f_ratios = list(jnp.linspace(*foreground_ratios, depth))
        else:
            raise ValueError("Unknown type for forefround_ratios, got:")

        n_chunks = len(depths)
        num_heads = to_list(num_heads, n_chunks)
        self.num_features = int(dim * 2 ** (n_chunks - 1))
        self.blocks = [
            BlockChunk(
                block=block,
                repeat=repeat,
                dim=int(dim * 2**i),
                depth=depths[i],
                use_cpe=False,
                downsampler=PatchMerging if (i < n_chunks - 1) else eqx.nn.Identity,
                downsampler_contains_dropout=False,
                foreground_ratio=f_ratios[i],
                patch_size=patch_size,
                num_heads=num_heads[i],
                head_expand_ratio=head_expand_ratio,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                proj_ratio=proj_ratio,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                init_values=init_values,
                key=block_subkeys[i],
            )
            for i, depth in enumerate(depths)
        ]

        self.norm = norm_layer(self.num_features)
        self.head = (
            eqx.nn.Linear(self.num_features, num_classes, key=key_head)
            if num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        return_qa: bool = False,
    ) -> Float[Array, "seqlen dim"]:
        """Extract features from input image using partial attention.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Tuple of (processed features, final query attention token)
        """
        key_posdrop, *block_subkeys = jr.split(key, len(self.blocks) + 1)
        x = self.patch_embed(x)
        x = self.pos_drop(x, inference=inference, key=key_posdrop)

        qa = self.qa_token
        for blk, key_block in zip(self.blocks, block_subkeys):
            x, qa = blk(
                x,
                qa=qa,
                inference=inference,
                key=key_block,
            )

        if return_qa:
            return x, qa
        return x

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "num_classes"]:
        """Process input image through the full network.

        Args:
            x: Input image tensor
            inference: Whether to enable dropout during inference
            key: PRNG key for random operations

        Returns:
            Classification logits for each class
        """
        x = self.features(x, inference=inference, key=key)
        x = jax.vmap(self.norm)(x)
        x = reduce(x, "n c -> c", "mean")

        x = self.head(x)

        return x
