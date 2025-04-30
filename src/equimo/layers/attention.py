from typing import Callable, List, Literal, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange, reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.convolution import ConvBlock, SingleConvBlock, MBConv
from equimo.layers.dropout import DropPathAdd
from equimo.layers.ffn import Mlp
from equimo.layers.mamba import Mamba2Mixer
from equimo.layers.norm import LayerScale
from equimo.layers.posemb import PosCNN2D, PosEmbMLPSwinv1D, PosEmbMLPSwinv2D, RoPE
from equimo.utils import nearest_power_of_2_divisor


class Attention(eqx.Module):
    """Multi-head self attention module.

    A standard transformer-style attention implementation with query, key and value
    projections followed by scaled dot-product attention.

    Attributes:
        dim: Total dimension of the input/output
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head (dim // num_heads)
    """

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    q_norm: eqx.Module
    k_norm: eqx.Module
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        eps: float = 1e-5,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        key_qkv, key_proj = jr.split(key, 2)
        self.qkv = eqx.nn.Linear(dim, dim * 3, use_bias=qkv_bias, key=key_qkv)
        self.proj = eqx.nn.Linear(dim, dim, use_bias=proj_bias, key=key_proj)

        self.q_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()
        self.k_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()

        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        mask: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "seqlen dim"]:
        key1, key2 = jr.split(key, 2)

        qkv = jax.vmap(self.qkv)(x)
        qkv = rearrange(
            qkv,
            "s (n h d) -> n h s d",
            n=3,
            h=self.num_heads,
            d=self.dim // self.num_heads,
        )
        q, k, v = qkv
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)

        attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)

        if mask is not None:
            attn = jnp.where(mask == 0, jnp.finfo(attn.dtype).min, attn)

        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, inference=inference, key=key1)

        x = jnp.einsum("hqk,hkd->hqd", attn, v)
        x = rearrange(x, "h s d -> s (h d)")
        x = jax.vmap(self.proj)(x)
        x = self.proj_drop(x, inference=inference, key=key2)

        return x


class WindowedAttention(eqx.Module):
    """Windowed multi-head self attention module.

    Applies self-attention within local windows of the input sequence.
    Includes relative position embeddings within each window.

    Attributes:
        dim: Total dimension of the input/output
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head (dim // num_heads)
        resolution: Size of each attention window (window_size)
    """

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    resolution: int = eqx.field(static=True)

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    q_norm: eqx.Module
    k_norm: eqx.Module
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout
    pos_emb_funct: PosEmbMLPSwinv2D

    def __init__(
        self,
        dim: int,
        num_heads: int,
        resolution: int,
        seq_len: int,
        *,
        key: PRNGKeyArray,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        eps: float = 1e-5,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.resolution = resolution

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        key_qkv, key_proj, key_posemb = jr.split(key, 3)
        self.qkv = eqx.nn.Linear(dim, dim * 3, use_bias=qkv_bias, key=key_qkv)
        self.proj = eqx.nn.Linear(dim, dim, use_bias=proj_bias, key=key_proj)

        self.pos_emb_funct = PosEmbMLPSwinv2D(
            window_size=[resolution, resolution],
            pretrained_window_size=[resolution, resolution],
            num_heads=num_heads,
            seq_len=seq_len,
            key=key_posemb,
        )

        self.q_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()
        self.k_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()

        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key1, key2 = jr.split(key, 2)

        qkv = jax.vmap(self.qkv)(x)
        qkv = rearrange(
            qkv,
            "s (n h d) -> n h s d",
            n=3,
            h=self.num_heads,
            d=self.dim // self.num_heads,
        )
        q, k, v = qkv
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)

        attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
        attn = self.pos_emb_funct(attn, self.resolution**2)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, inference=inference, key=key1)

        x = jnp.einsum("hqk,hkd->hqd", attn, v)
        x = rearrange(x, "h s d -> s (h d)")
        x = jax.vmap(self.proj)(x)
        x = self.proj_drop(x, inference=inference, key=key2)

        return x


class AttentionBlock(eqx.Module):
    """Standard transformer block with attention and MLP.

    Implements a full transformer block with:
    - Multi-head self attention
    - Layer normalization
    - MLP feed-forward network
    - Residual connections
    - Optional layer scaling
    - eqx.nn.Dropout paths

    Attributes:
        prenorm: First layer normalization (before attention)
        postnorm: Second layer normalization (optional, after attention)
        norm: Third layer normalization (after residual)
        ls1: First layer scale (optional)
        ls2: Second layer scale (optional)
    """

    prenorm: eqx.Module
    postnorm: eqx.Module
    norm: eqx.Module
    ls1: LayerScale
    ls2: LayerScale
    attn: eqx.Module
    mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 4.0,
        drop_path: float | List[float] = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = jax.nn.gelu,
        attn_layer: eqx.Module = Attention,
        ffn_layer: eqx.Module = Mlp,
        ffn_bias: bool = True,
        ffn_norm: bool = False,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        post_attention_norm: bool = False,
        init_values: float | None = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        key_attn, key_mlp = jr.split(key, 2)

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)} ({drop_path})."
                )
            dr1, dr2 = drop_path
            dr1 = float(dr1)
            dr2 = float(dr2)
        else:
            dr1 = dr2 = float(drop_path)

        self.prenorm = norm_layer(dim, eps=eps)
        self.postnorm = (
            norm_layer(dim, eps=eps) if post_attention_norm else eqx.nn.Identity()
        )
        self.norm = norm_layer(dim, eps=eps)

        if init_values:
            self.ls1 = LayerScale(dim, init_values=init_values)
            self.ls2 = LayerScale(dim, init_values=init_values)
        else:
            self.ls1 = self.ls2 = eqx.nn.Identity()

        self.attn = attn_layer(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            eps=eps,
            key=key_attn,
        )

        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if ffn_norm else None,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            eps=eps,
            key=key_mlp,
        )

        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        mask: Optional[Float[Array, ""]] = None,
    ) -> Float[Array, "seqlen dim"]:
        key_attn, key_mlp, key_dr1, key_dr2 = jr.split(key, 4)

        # I chose to define extra args here rather than passing mask directly
        # because not all attention mechanisms support masks as args
        extra_kwargs = {"mask": mask} if mask is not None else {}

        x = self.drop_path1(
            x,
            self.ls1(
                jax.vmap(self.postnorm)(
                    self.attn(
                        jax.vmap(self.prenorm)(x),
                        inference=inference,
                        key=key_attn,
                        **extra_kwargs,
                    )
                )
            ),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.ls2(
                self.mlp(
                    jax.vmap(self.norm)(x),
                    inference=inference,
                    key=key_mlp,
                    **extra_kwargs,
                )
            ),
            inference=inference,
            key=key_dr2,
        )

        return x


class HATBlock(eqx.Module):
    """Hierarchical Attention Transformer block.

    Implements hierarchical attention with:
    - Local window attention
    - Carrier tokens for global interaction
    - Optional token propagation in final layer
    - Hierarchical structure with different spatial resolutions

    Attributes:
        do_propagation: Whether to propagate carrier token info in last layer
        sr_ratio: Spatial reduction ratio for hierarchical structure
        window_size: Size of local attention windows
        ct_size: Size of carrier token grid
        last: Whether this is the last block in the network
    """

    do_propagation: bool = eqx.field(static=True)
    sr_ratio: float = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    ct_size: int = eqx.field(static=True)
    last: bool = eqx.field(static=True)

    norm1: eqx.Module
    norm2: eqx.Module
    ls1: LayerScale
    ls2: LayerScale
    attn: eqx.Module
    mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd
    pos_embed: PosEmbMLPSwinv1D

    hat_norm1: Optional[eqx.Module] = eqx.field(default=None)
    hat_norm2: Optional[eqx.Module] = eqx.field(default=None)
    hat_norm3: Optional[eqx.Module] = eqx.field(default=None)
    hat_attn: Optional[eqx.Module] = eqx.field(default=None)
    hat_mlp: Optional[eqx.Module] = eqx.field(default=None)
    hat_drop_path: Optional[DropPathAdd] = eqx.field(default=None)
    hat_pos_embed: Optional[PosEmbMLPSwinv1D] = eqx.field(default=None)
    hat_ls1: Optional[LayerScale] = eqx.field(default=None)
    hat_ls2: Optional[LayerScale] = eqx.field(default=None)
    hat_ls3: Optional[LayerScale] = eqx.field(default=None)

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 4.0,
        drop_path: float | List[float] = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Callable = jax.nn.gelu,
        attn_layer: eqx.Module = Attention,
        ffn_layer: eqx.Module = Mlp,
        ffn_bias: bool = True,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        init_values: float | None = None,
        sr_ratio: float = 1.0,
        ct_size: int = 1,
        last: bool = False,
        do_propagation: bool = False,
        eps: float = 1e-5,
        **kwargs,
    ):
        key_posemb, key_hatposemb, key_attn, key_hatattn, key_mlp, key_hatmlp = (
            jr.split(key, 6)
        )
        self.do_propagation = do_propagation
        self.sr_ratio = sr_ratio
        self.window_size = window_size
        self.ct_size = ct_size
        self.last = last

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)} ({drop_path})."
                )
            dr1, dr2 = drop_path
            dr1 = float(dr1)
            dr2 = float(dr2)
        else:
            dr1 = dr2 = float(drop_path)

        self.pos_embed = PosEmbMLPSwinv1D(
            dim, rank=2, seq_len=window_size**2, key=key_posemb
        )
        self.norm1 = norm_layer(dim, eps=eps)
        self.norm2 = norm_layer(dim, eps=eps)

        if init_values:
            self.ls1 = LayerScale(dim, init_values=init_values)
            self.ls2 = LayerScale(dim, init_values=init_values)
        else:
            self.ls1 = self.ls2 = eqx.nn.Identity()

        # number of carrier tokens per every window
        cr_tokens_per_window = self.ct_size**2 if self.sr_ratio > 1 else 0
        cr_tokens_total = cr_tokens_per_window * self.sr_ratio * self.sr_ratio

        self.attn = WindowedAttention(
            resolution=self.window_size,
            seq_len=self.window_size**2 + cr_tokens_per_window,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            eps=eps,
            key=key_attn,
        )

        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            eps=eps,
            key=key_mlp,
        )

        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

        if self.sr_ratio > 1:
            # If hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim, eps=eps)
            self.hat_norm2 = norm_layer(dim, eps=eps)

            self.hat_attn = WindowedAttention(
                resolution=int(cr_tokens_total**0.5),
                seq_len=cr_tokens_total,
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                eps=eps,
                key=key_hatattn,
            )

            self.hat_mlp = ffn_layer(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                dropout_rate=proj_drop,
                bias=ffn_bias,
                eps=eps,
                key=key_hatmlp,
            )

            self.hat_drop_path = DropPathAdd(dr2)

            self.hat_pos_embed = PosEmbMLPSwinv1D(
                dim, rank=2, seq_len=cr_tokens_total, key=key_hatposemb
            )
            self.hat_ls1 = (
                LayerScale(dim, init_values=init_values)
                if init_values
                else eqx.nn.Identity()
            )
            self.hat_ls2 = (
                LayerScale(dim, init_values=init_values)
                if init_values
                else eqx.nn.Identity()
            )
            self.hat_ls3 = (
                LayerScale(dim, init_values=init_values)
                if init_values
                else eqx.nn.Identity()
            )

    def ct_window(self, ct, H, W, window_size):
        return rearrange(
            ct,
            "(h h1 w w1) d -> (h w h1 w1) d",
            h=H,
            w=W,
            h1=window_size,
            w1=window_size,
        )

    def ct_dewindow(self, ct, H, W, window_size):
        return rearrange(
            ct,
            "(h w h1 w1) d -> (h h1 w w1) d",
            h=H,
            w=W,
            h1=window_size,
            w1=window_size,
        )

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        carrier_tokens: Float[Array, "..."],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key_attn, key_hattn, key_dr1, key_hdr1, key_dr2, key_hdr2, key_mlp, key_hmlp = (
            jr.split(key, 8)
        )

        s, n = x.shape
        x = self.pos_embed(x)
        ct = carrier_tokens

        if self.sr_ratio > 1:
            # do hierarchical attention via carrier tokens
            # first do attention for carrier tokens
            ng, hg = ct.shape

            # ct are located quite differently
            ct = self.ct_dewindow(
                ct,
                self.sr_ratio,
                self.sr_ratio,
                self.ct_size,
            )

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)

            # attention plus mlp
            ct = self.hat_drop_path(
                ct,
                self.hat_ls1(
                    self.hat_attn(
                        jax.vmap(self.hat_norm1)(ct),
                        inference=inference,
                        key=key_hattn,
                    )
                ),
                inference=inference,
                key=key_hdr1,
            )
            ct = self.hat_drop_path(
                ct,
                self.hat_ls2(
                    self.hat_mlp(
                        jax.vmap(self.hat_norm2)(ct),
                        inference=inference,
                        key=key_hmlp,
                    )
                ),
                inference=inference,
                key=key_hdr2,
            )

            # ct are put back to windows
            ct = self.ct_window(
                ct,
                self.sr_ratio,
                self.sr_ratio,
                self.ct_size,
            )

            # concatenate carrier_tokens to the windowed tokens
            x = jnp.concatenate((x, ct), axis=0)

        x = self.drop_path1(
            x,
            self.ls1(
                self.attn(
                    jax.vmap(self.norm1)(x),
                    inference=inference,
                    key=key_attn,
                )
            ),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.ls2(
                self.mlp(
                    jax.vmap(self.norm2)(x),
                    inference=inference,
                    key=key_mlp,
                )
            ),
            inference=inference,
            key=key_dr2,
        )

        if self.sr_ratio > 1:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            split_index = self.window_size * self.window_size
            x, ctr = jnp.split(x, [split_index], axis=0)

            if self.last and self.do_propagation:
                # propagate carrier token information into the image
                ctr_image_space = rearrange(
                    ctr,
                    "(h w) c -> c h w",
                    h=self.ct_size * self.sr_ratio,
                    w=self.ct_size * self.sr_ratio,
                )
                upsampled = jax.image.resize(
                    ctr_image_space,
                    (n, self.window_size, self.window_size),
                    method="nearest",
                )
                upsampled = rearrange(upsampled, "c h w -> (h w) c")

                x = x + self.hat_ls3(upsampled)

        return x, ct


class SHSA(eqx.Module):
    """Signe-Head Self Attention module from the SHViT paper.

    "In SHSA, self-attention with a single head is applied to just a subset of
    the input channels, while the others remain unchanged. SHSA layer not only
    eliminates the computational redundancy derived from multi-head mechanism
    but also reduces memory access cost by processing partial channels"

    Attributes:
        scale: Scaling factor for attention scores
        qk_dim: Dimension of query/key projections
        pdim: Dimension of primary features
    """

    scale: eqx.field(static=True)
    qk_dim: eqx.field(static=True)
    pdim: eqx.field(static=True)

    pre_norm: eqx.Module
    qkv: eqx.Module
    proj: eqx.Module

    def __init__(
        self,
        dim: int,
        qk_dim: int,
        pdim: int,
        *,
        key: PRNGKeyArray,
        norm_max_group: int = 32,
        **kwargs,
    ):
        key_conv1, key_conv2 = jr.split(key, 2)

        self.scale = qk_dim**-0.5
        self.qk_dim = qk_dim
        self.pdim = pdim

        self.pre_norm = eqx.nn.GroupNorm(1, pdim)
        self.qkv = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=pdim,
            out_channels=qk_dim * 2 + pdim,
            kernel_size=1,
            stride=1,
            padding=0,
            key=key_conv1,
        )
        num_groups = nearest_power_of_2_divisor(dim, norm_max_group)
        self.proj = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv(
                    num_spatial_dims=2,
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    key=key_conv2,
                ),
                eqx.nn.GroupNorm(num_groups, dim),
            ]
        )

    def flatten(self, x: Float[Array, "channels height width"]):
        return rearrange(x, "c h w -> c (h w)")

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        inference: Optional[bool] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "channels height width"]:
        C, H, W = x.shape
        x1, x2 = jnp.split(x, [self.pdim], axis=0)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = jnp.split(qkv, [self.qk_dim, self.qk_dim * 2], axis=0)
        q, k, v = self.flatten(q), self.flatten(k), self.flatten(v)

        attn = jnp.einsum("dq,dk->qk", q, k) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)

        # TODO: verify einsum, qk,kd->qd
        x1 = jnp.einsum("qk,dv->qd", attn, v)
        x1 = rearrange(x1, "(h w) d -> d h w", h=H, w=W, d=self.pdim)

        x = jnp.concat([x1, x2], axis=0)
        x = self.proj(x)

        return x


class LinearAttention(eqx.Module):
    """Linear attention with rotary position encoding.

    Implements efficient linear attention with:
    - Linear complexity in sequence length
    - Rotary position embeddings
    - Learnable position encoding (LePE)
    - Multi-head structure

    Attributes:
        num_heads: Number of attention heads
    """

    num_heads: int = eqx.field(static=True)

    qk: eqx.Module
    lepe: eqx.Module
    rope: eqx.Module

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_fc1, key_conv = jr.split(key, 2)
        self.num_heads = num_heads

        self.qk = eqx.nn.Linear(dim, dim * 2, key=key_fc1)
        self.lepe = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            key=key_conv,
        )
        self.rope = RoPE((input_resolution[0], input_resolution[1], dim))

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        inference: Optional[bool] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seqlen dim"]:
        n, c = x.shape
        h = w = int(n**0.5)

        q, k = rearrange(
            jax.vmap(self.qk)(x),
            "n (qk h d) -> qk h n d",
            qk=2,
            h=self.num_heads,
        )
        v = rearrange(x, "n (h d) -> h n d", h=self.num_heads)

        q = jax.nn.elu(q) + 1.0
        k = jax.nn.elu(k) + 1.0

        q_2d = rearrange(q, "h (x y) d -> x y (h d)", x=h, y=w)
        k_2d = rearrange(k, "h (x y) d -> x y (h d)", x=h, y=w)

        q_rope = rearrange(self.rope(q_2d), "x y (h d) -> h (x y) d", h=self.num_heads)
        k_rope = rearrange(self.rope(k_2d), "x y (h d) -> h (x y) d", h=self.num_heads)

        # Compute attention
        z = 1 / (jnp.einsum("hnd,hd->hn", q, reduce(k, "h n d -> h d", "mean")) + 1e-6)
        kv = jnp.einsum("hnd,hne->hde", k_rope * (n**-0.5), v * (n**-0.5))
        x = jnp.einsum("hnd,hde->hne", q_rope, kv) * z[..., None]

        # Reshape output
        x = rearrange(x, "h n d -> n (h d)")

        # Apply LePE
        v_2d = rearrange(v, "h (x y) d -> (h d) x y", x=h, y=w)
        lepe_out = self.lepe(v_2d)

        lepe_out = rearrange(lepe_out, "(h d) x y -> (x y) (h d)", h=self.num_heads)

        # Combine attention output and LePE
        x = x + lepe_out

        return x


class MllaBlock(eqx.Module):
    """Mamba-like Linear Attention block.

    Implements a transformer block using linear attention that:
    - Uses convolutional position encoding
    - Optionally includes depthwise convolution
    - Has residual connections and layer norms
    - Includes MLP feed-forward network

    Attributes:
        use_dwc: Whether to use depthwise convolution
    """

    use_dwc: bool = eqx.field(static=True)

    act: Callable
    cpe1: eqx.Module
    cpe2: eqx.Module
    norm1: eqx.Module
    norm2: eqx.Module
    in_proj: eqx.Module
    act_proj: eqx.Module
    out_proj: eqx.Module
    dwc: eqx.Module
    attn: eqx.Module
    drop_path1: eqx.Module
    drop_path2: eqx.Module
    mlp: eqx.Module

    def __init__(
        self,
        dim: int,
        *,
        key: PRNGKeyArray,
        act_layer: Callable = jax.nn.silu,  # gelu in VSSD
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        use_dwc: bool = True,  # For Mlla but not VMamba-2
        attention_layer: eqx.Module = LinearAttention,
        drop_path: List[float] | float = 0.0,
        mlp_ratio: float = 4.0,
        ffn_layer: eqx.Module = Mlp,
        ffn_bias: bool = True,
        proj_drop: float = 0.0,
        eps: float = 1e-5,
        **kwargs,
    ):
        if attention_layer not in [Attention, LinearAttention, Mamba2Mixer]:
            raise ValueError(
                "Unsupported `attention_layer`, got:",
                attention_layer,
            )

        (
            key_conv1,
            key_conv2,
            key_conv3,
            key_fc1,
            key_fc2,
            key_fc3,
            key_attn,
            key_mlp,
        ) = jr.split(key, 8)
        self.use_dwc = use_dwc
        self.act = act_layer

        self.cpe1 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            use_bias=True,
            key=key_conv1,
        )
        self.norm1 = norm_layer(dim, eps=eps)

        if use_dwc:
            self.in_proj = eqx.nn.Linear(dim, dim, key=key_fc1)
            self.act_proj = eqx.nn.Linear(dim, dim, key=key_fc2)
            self.dwc = eqx.nn.Conv(
                num_spatial_dims=2,
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim,
                use_bias=True,
                key=key_conv2,
            )
            self.out_proj = eqx.nn.Linear(dim, dim, key=key_fc3)
        else:
            self.in_proj = self.act_proj = self.dwc = self.out_proj = eqx.nn.Identity()

        config = {"d_model": dim} if attention_layer is Mamba2Mixer else {"dim": dim}

        self.attn = attention_layer(
            **(kwargs | config),
            key=key_attn,
        )

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)} ({drop_path})."
                )
            dr1, dr2 = drop_path
            dr1 = float(dr1)
            dr2 = float(dr2)
        else:
            dr1 = dr2 = float(drop_path)

        self.drop_path1 = DropPathAdd(dr1)

        self.cpe2 = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            use_bias=True,
            key=key_conv3,
        )

        self.norm2 = norm_layer(dim, eps=eps)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            key=key_mlp,
        )

        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        inference: Optional[bool] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seqlen dim"]:
        key_attn, key_dr1, key_dr2, key_mlp = jr.split(key, 4)
        l, _ = x.shape
        h = w = int(l**0.5)

        x1 = x + rearrange(
            self.cpe1(rearrange(x, "(h w) c -> c h w", h=h, w=w)),
            "c h w -> (h w) c",
        )
        x1 = jax.vmap(self.norm1)(x1)

        if self.use_dwc:
            act_res = self.act(jax.vmap(self.act_proj)(x1))

            x1 = rearrange(jax.vmap(self.in_proj)(x1), "(h w) c -> c h w", h=h, w=w)
            x1 = self.act(rearrange(self.dwc(x1), "c h w -> (h w) c"))

        x1 = self.attn(x1, inference=inference, key=key_attn)

        if self.use_dwc:
            x1 = jax.vmap(self.out_proj)(x * act_res)

        x = self.drop_path1(x, x1, inference=inference, key=key_dr1)

        x += rearrange(
            self.cpe2(rearrange(x, "(h w) c -> c h w", h=h, w=w)),
            "c h w -> (h w) c",
        )

        return self.drop_path2(
            x,
            self.mlp(jax.vmap(self.norm2)(x), inference=inference, key=key_mlp),
            inference=inference,
            key=key_dr2,
        )


class MMSA(eqx.Module):
    """Mixed Multi-head Self Attention.

    Implements attention with:
    - Multi-head structure
    - Attention score projection for multi-scale interaction
    - Normalized query/key processing
    - eqx.nn.Dropout regularization

    Attributes:
        dim: Total dimension of input/output
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
    """

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_proj1: eqx.nn.Linear
    attn_proj2: eqx.nn.Linear
    q_norm: eqx.Module
    k_norm: eqx.Module
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        head_expand_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        eps: float = 1e-5,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        key_qkv, key_proj, key_attnproj1, key_attnproj2 = jr.split(key, 4)
        self.qkv = eqx.nn.Linear(dim, dim * 3, use_bias=qkv_bias, key=key_qkv)
        self.proj = eqx.nn.Linear(dim, dim, use_bias=proj_bias, key=key_proj)
        self.attn_proj1 = eqx.nn.Linear(
            num_heads,
            int(num_heads * head_expand_ratio),
            key=key_attnproj1,
        )
        self.attn_proj2 = eqx.nn.Linear(
            int(num_heads * head_expand_ratio),
            num_heads,
            key=key_attnproj2,
        )

        self.q_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()
        self.k_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()

        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key1, key2 = jr.split(key, 2)

        qkv = jax.vmap(self.qkv)(x)
        qkv = rearrange(
            qkv,
            "s (n h d) -> n h s d",
            n=3,
            h=self.num_heads,
            d=self.dim // self.num_heads,
        )
        q, k, v = qkv
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)

        attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
        attn = jax.vmap(jax.vmap(self.attn_proj1))(
            rearrange(attn, "h q k -> q k h"),
        )
        attn = jax.nn.softmax(attn, axis=1)
        attn = rearrange(
            jax.vmap(jax.vmap(self.attn_proj2))(attn),
            "q k h -> h q k",
        )
        attn = self.attn_drop(attn, inference=inference, key=key1)

        x = jnp.einsum("hqk,hkd->hqd", attn, v)
        x = rearrange(x, "h s d -> s (h d)")
        x = jax.vmap(self.proj)(x)
        x = self.proj_drop(x, inference=inference, key=key2)

        return x


class SQA(eqx.Module):
    """Single Query Attention module.

    Implements efficient attention where:
    - Only one query is used against many keys/values
    - Includes projection layers for dimension reduction
    - Uses normalized attention computation
    - Includes residual connection

    Attributes:
        dim: Total dimension of input/output
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
    """

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    kv: eqx.nn.Linear
    proj1: eqx.nn.Linear
    proj2: eqx.nn.Linear
    proj_norm: eqx.Module
    q_norm: eqx.Module
    k_norm: eqx.Module
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        key: PRNGKeyArray,
        kv_bias: bool = True,
        proj_bias: bool = True,
        proj_ratio: float = 4.0,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        eps: float = 1e-5,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        key_kv, key_proj = jr.split(key, 2)
        self.kv = eqx.nn.Linear(dim, dim * 2, use_bias=kv_bias, key=key_kv)
        self.proj1 = eqx.nn.Linear(
            dim, int(dim // proj_ratio), use_bias=proj_bias, key=key_proj
        )
        self.proj2 = eqx.nn.Linear(
            int(dim // proj_ratio), dim, use_bias=proj_bias, key=key_proj
        )
        self.proj_norm = norm_layer(int(dim // proj_ratio), eps=eps)

        self.q_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()
        self.k_norm = norm_layer(dim, eps=eps) if qk_norm else eqx.nn.Identity()

        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        q: Float[Array, "1 dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen_x dim"]:
        key1, key2 = jr.split(key, 2)

        q = rearrange(
            q,
            "1 (h d) -> h 1 d",
            h=self.num_heads,
            d=self.dim // self.num_heads,
        )
        kv = jax.vmap(self.kv)(x)
        kv = rearrange(
            kv,
            "s (n h d) -> n h s d",
            n=2,
            h=self.num_heads,
            d=self.dim // self.num_heads,
        )
        k, v = kv
        q = jax.vmap(jax.vmap(self.q_norm))(q)
        k = jax.vmap(jax.vmap(self.k_norm))(k)

        attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, inference=inference, key=key1)

        x1 = jnp.einsum("hqk,hkd->hqd", attn, v)
        x1 = rearrange(x1, "h s d -> s (h d)")
        x1 = jax.vmap(self.proj_norm)(jax.vmap(self.proj1)(x1))
        x1 = jax.vmap(self.proj2)(jax.nn.relu(x1))

        x = self.proj_drop(x + x1, inference=inference, key=key2)

        return x


class PartialFormerBlock(eqx.Module):
    """Partial Transformer block with foreground/background separation.

    Implements a specialized transformer that:
    - Separates input into foreground/background regions
    - Uses different attention mechanisms for each region
    - Includes positional encoding and MLP layers
    - Has layer scaling and dropout paths

    Attributes:
        foreground_ratio: Ratio of tokens treated as foreground
        patch_size: Size of input patches
    """

    foreground_ratio: float = eqx.field(static=True)
    patch_size: float = eqx.field(static=True)

    act: Callable
    posemb: eqx.Module
    norm1: eqx.Module
    norm2: eqx.Module
    mmsa: eqx.Module
    sqa: eqx.Module
    ls1: eqx.Module
    ls2: eqx.Module
    drop_path1: eqx.Module
    drop_path2: eqx.Module
    mlp: eqx.Module

    def __init__(
        self,
        dim: int,
        num_heads: int,
        foreground_ratio: int,
        patch_size: int,
        *,
        key: PRNGKeyArray,
        head_expand_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_ratio: float = 4.0,
        act_layer: Callable = jax.nn.silu,  # gelu in VSSD
        norm_layer: eqx.Module = eqx.nn.LayerNorm,
        drop_path: List[float] | float = 0.0,
        mlp_ratio: float = 4.0,
        ffn_layer: eqx.Module = Mlp,
        ffn_bias: bool = True,
        init_values: float | None = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        (
            key_posemb,
            key_mmsa,
            key_sqa,
            key_mlp,
        ) = jr.split(key, 4)
        self.act = act_layer
        self.foreground_ratio = foreground_ratio
        self.patch_size = patch_size

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)} ({drop_path})."
                )
            dr1, dr2 = drop_path
            dr1 = float(dr1)
            dr2 = float(dr2)
        else:
            dr1 = dr2 = float(drop_path)

        self.norm1 = norm_layer(dim, eps=eps)
        self.norm2 = norm_layer(dim, eps=eps)

        if init_values:
            self.ls1 = LayerScale(dim, init_values=init_values)
            self.ls2 = LayerScale(dim, init_values=init_values)
        else:
            self.ls1 = self.ls2 = eqx.nn.Identity()

        self.posemb = PosCNN2D(dim, dim, key=key_posemb)

        self.mmsa = MMSA(
            dim=dim,
            num_heads=num_heads,
            head_expand_ratio=head_expand_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            eps=eps,
            key=key_mmsa,
        )
        self.sqa = SQA(
            dim=dim,
            num_heads=num_heads,
            kv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_bias=proj_bias,
            proj_ratio=proj_ratio,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            eps=eps,
            key=key_sqa,
        )

        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            dropout_rate=proj_drop,
            bias=ffn_bias,
            eps=eps,
            key=key_mlp,
        )

        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        qa: Float[Array, "1 dim"],
        inference: Optional[bool] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[Float[Array, "seqlen dim"], Float[Array, "1 dim"]]:
        key_mmsa, key_sqa, key_dr1, key_dr2, key_mlp = jr.split(key, 5)
        l, _ = x.shape
        h = w = int(l**0.5)

        x1 = rearrange(x, "(h w) c -> c h w", h=h, w=w)
        x1 = self.posemb(x1)
        x1 = rearrange(
            x1,
            "c (h h1) (w w1) -> (h w) (h1 w1) c",
            h1=self.patch_size,
            w1=self.patch_size,
        )

        n = x1.shape[0]
        nf = int(n * self.foreground_ratio)

        idx = jnp.argsort(reduce(x1, "n s c -> n", "mean"), descending=True)
        nf_idx, nb_idx = jnp.split(idx, [nf])

        f, b = x1[nf_idx], x1[nb_idx]

        f = rearrange(f, "n s c -> (n s) c")
        b = rearrange(b, "n s c -> (n s) c")

        qf = self.mmsa(jnp.concat([qa, f], axis=0), inference=inference, key=key_mmsa)
        qa, f = jnp.split(qf, [1])
        b = self.sqa(b, qa, inference=inference, key=key_sqa)

        x1 = self.ls1(jnp.concat([f, b], axis=0))
        x = self.drop_path1(x, x1, inference=inference, key=key_dr1)

        qa, x1 = jnp.split(
            self.mlp(
                jax.vmap(self.norm2)(jnp.concat([qa, x])),
                inference=inference,
                key=key_mlp,
            )[1],
        )
        x = self.drop_path2(
            x,
            self.ls2(x1),
            inference=inference,
            key=key_dr2,
        )

        return x, qa


class LinearAngularAttention(eqx.Module):
    """Linear Angular Attention with optional sparsity.

    This is the base attention module to be included in a ViT to replicate
    Castling-ViT[1].
    Implements an efficient attention variant that:
    - Uses normalized vectors for computing attention
    - Applies optional sparsity regularization
    - Includes depthwise convolution for local mixing
    - Has linear complexity in sequence length

    Attributes:
        dim: Total dimension of input/output
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        sparse_reg: Whether to use sparsity regularization
        sparsity_threshold: Threshold for sparse attention

    References:
        [1]: You, et al., 2024. https://arxiv.org/abs/2211.10526
    """

    dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    sparse_reg: bool = eqx.field(static=True)
    sparsity_threshold: float = eqx.field(static=True)

    qkv: eqx.nn.Linear
    proj: eqx.nn.Linear
    attn_drop: eqx.nn.Dropout
    proj_drop: eqx.nn.Dropout
    dconv: eqx.nn.Conv

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sparse_reg: bool = False,
        *,
        key: PRNGKeyArray,
        sparsity_threshold: float = 0.2,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        res_kernel_size: int = 9,
        **kwargs,
    ):
        self.sparse_reg = sparse_reg
        self.sparsity_threshold = sparsity_threshold
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        key_qkv, key_proj, key_dconv = jr.split(key, 3)
        self.qkv = eqx.nn.Linear(dim, dim * 3, use_bias=qkv_bias, key=key_qkv)
        self.proj = eqx.nn.Linear(dim, dim, use_bias=proj_bias, key=key_proj)

        self.dconv = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=num_heads,
            out_channels=num_heads,
            kernel_size=(res_kernel_size, 1),
            stride=1,
            padding=(res_kernel_size // 2, 0),
            groups=num_heads,
            use_bias=False,
            key=key_dconv,
        )

        self.attn_drop = eqx.nn.Dropout(attn_drop)
        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key1, key2 = jr.split(key, 2)

        qkv = jax.vmap(self.qkv)(x)
        qkv = rearrange(
            qkv,
            "s (n h d) -> n h s d",
            n=3,
            h=self.num_heads,
            d=self.dim // self.num_heads,
        )
        q, k, v = qkv

        if self.sparse_reg:
            attn = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.head_dim)
            attn = jax.nn.softmax(attn, axis=-1)
            attn = self.attn_drop(attn, inference=inference, key=key1)
            sparse = jnp.where(attn > self.sparsity_threshold, attn, 0)

        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        k = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
        dconv_v = self.dconv(v)

        attn = jnp.einsum("hqk,hqv->hkv", k, v)

        if self.sparse_reg:
            x = (sparse @ v) + 0.5 * v + 1.0 / jnp.pi * (q @ attn)
        else:
            x = 0.5 * v + 1.0 / jnp.pi * (q @ attn)

        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        x += dconv_v

        x = rearrange(x, "h s d -> s (h d)")
        x = jax.vmap(self.proj)(x)
        x = self.proj_drop(x, inference=inference, key=key2)

        return x


class RFAttention(eqx.Module):
    """Attention with Tensor Reduction by Summation[1].

    A ReLU Linear Attention mechanism replacing matmuls with global
    summation and element-wise multiplications.

    Attributes:
        dim: Total dimension of the input/output
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head (dim // num_heads)

    References:
        [1]. Yang, J., An, L., & Park, S. I. (2024). ReduceFormer: Attention
            with Tensor Reduction by Summation (No. arXiv:2406.07488). arXiv.
            https://doi.org/10.48550/arXiv.2406.07488
    """

    total_dim: int = eqx.field(static=True)
    kernel_func: Callable = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    qkv: eqx.nn.Conv2d
    aggreg: list[eqx.nn.Conv2d]
    proj: SingleConvBlock
    proj_drop: eqx.nn.Dropout

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        num_heads: int | None = None,
        head_dim: int = 8,
        heads_ratio: float = 1.0,
        scales: Sequence[int] = (5,),
        use_bias: bool = False,
        kernel_func: Callable = jax.nn.relu,
        proj_drop: float = 0.0,
        # TODO: Benchmark against LN, RMSN, NsLN
        norm_layer: eqx.Module = eqx.nn.GroupNorm,
        norm_kwargs: dict = {},
        eps: float = 1e-15,
        **kwargs,
    ):
        key_qkv, key_aggreg, key_proj = jr.split(key, 3)

        self.kernel_func = kernel_func
        self.eps = eps
        num_heads = num_heads or int(in_channels // head_dim * heads_ratio)
        total_dim = num_heads * head_dim
        self.total_dim = total_dim * (1 + len(scales))

        self.qkv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=3 * total_dim,
            kernel_size=1,
            padding="SAME",
            use_bias=use_bias,
            key=key_qkv,
        )
        self.aggreg = [
            eqx.nn.Conv2d(
                in_channels=3 * total_dim,
                out_channels=3 * total_dim,
                kernel_size=scale,
                padding="SAME",
                groups=3 * total_dim,
                key=key_aggreg,
                use_bias=use_bias,
            )
            for scale in scales
        ]
        # TODO: test different normalizations
        self.proj = SingleConvBlock(
            in_channels=self.total_dim,
            out_channels=out_channels,
            kernel_size=1,
            use_bias=use_bias,
            norm_layer=norm_layer,
            norm_kwargs=norm_kwargs,
            key=key_proj,
        )

        self.proj_drop = eqx.nn.Dropout(proj_drop)

    def __call__(
        self,
        x: Float[Array, "dim height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "dim height width"]:
        qkv_base = self.qkv(x)

        aggregated_qkvs = [op(qkv_base) for op in self.aggreg]
        all_qkvs = [qkv_base] + aggregated_qkvs

        rearranged_qkvs = [
            rearrange(qkv, "(n d) h w -> n d h w", n=3) for qkv in all_qkvs
        ]
        multiscale_qkv = jnp.concatenate(rearranged_qkvs, axis=1)

        q, k, v = multiscale_qkv

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        sum_k = jnp.sum(k, axis=(-1, -2), keepdims=True)
        sum_v = jnp.sum(v * sum_k, axis=(-1, -2), keepdims=True)
        sum_kv = jnp.sum(k * sum_v, axis=(-1, -2), keepdims=True)
        sum_q = jnp.sum(q, axis=0, keepdims=True)

        out = (q * sum_kv) / (sum_q * sum_k + self.eps)
        out = self.proj(out)
        out = self.proj_drop(out, inference=inference, key=key)

        return out


class RFAttentionBlock(eqx.Module):
    context_module: RFAttention
    local_module: MBConv
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        in_channels: int,
        *,
        key,
        head_dim: int = 32,
        heads_ratio: float = 1.0,
        scales: Sequence[int] = (5,),
        rfattn_norm_layer: eqx.Module = eqx.nn.GroupNorm,
        norm_kwargs: dict = {},
        expand_ratio: float = 4.0,
        mbconv_norm_layers: tuple = (None, None, eqx.nn.GroupNorm),
        mbconv_act_layers: tuple = (jax.nn.hard_swish, jax.nn.hard_swish, None),
        fuse_mbconv: bool = False,
        context_drop: float = 0.0,
        local_drop: float = 0.0,
        drop_path: float | List[float] = 0.0,
        residual_mbconv: bool = False,
        **kwargs,
    ):
        key_context, key_local = jr.split(key, 2)

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)} ({drop_path})."
                )
            dr1, dr2 = drop_path
            dr1 = float(dr1)
            dr2 = float(dr2)
        else:
            dr1 = dr2 = float(drop_path)

        self.context_module = RFAttention(
            in_channels=in_channels,
            out_channels=in_channels,
            head_dim=head_dim,
            heads_ratio=heads_ratio,
            scales=scales,
            norm_layer=rfattn_norm_layer,
            norm_kwargs=norm_kwargs,
            proj_drop=context_drop,
            key=key_context,
        )
        self.local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            norm_layers=mbconv_norm_layers,
            act_layers=mbconv_act_layers,
            use_bias=(True, True, False),
            fuse=fuse_mbconv,
            dropout=local_drop,
            drop_path=dr1,
            residual=False,
            key=key_local,
        )

        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "dim height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ):
        key_context, key_local, key_dr1, key_dr2 = jr.split(key, 4)

        # TODO: some prenorm?
        x = self.drop_path1(
            x,
            self.context_module(x, inference=inference, key=key_context),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.local_module(x, inference=inference, key=key_local),
            inference=inference,
            key=key_dr2,
        )

        return x


class ConvAttention(eqx.Module):
    """Lightweight ConvAttention from LowFormer."""

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    attention_type: Literal["softmax", "sigmoid"] = eqx.field(static=True)

    qkv: eqx.nn.Sequential
    out_proj: eqx.nn.Conv2d | eqx.nn.Identity
    upsample: eqx.nn.ConvTranspose2d

    def __init__(
        self,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        att_kernel: int = 7,
        att_stride: int = 4,
        fuse: bool = True,
        attention_type: Literal["softmax", "sigmoid"] = "softmax",
        norm_layer: eqx.Module | None = eqx.nn.GroupNorm,
        norm_kwargs: dict = {},
        **kwargs,
    ):
        key_qkv1, key_qkv2, key_oproj, key_upsampling = jr.split(key, 4)

        self.attention_type = attention_type
        self.num_heads = int(max(1, in_channels // 30))
        self.head_dim = in_channels // self.num_heads
        total_dim = int(self.head_dim * self.num_heads * 3)

        self.qkv = eqx.nn.Sequential(
            [
                SingleConvBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=att_kernel,
                    stride=att_stride,
                    padding=att_kernel // 2,
                    groups=in_channels,
                    use_bias=False,
                    norm_layer=norm_layer,
                    act_layer=None,
                    key=key_qkv1,
                ),
                eqx.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=total_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    use_bias=False,
                    key=key_qkv2,
                ),
            ]
        )

        if fuse:
            self.out_proj = eqx.nn.Identity()
            self.upsample = eqx.nn.ConvTranspose2d(
                in_channels=self.head_dim * self.num_heads,
                out_channels=in_channels,
                kernel_size=3 if att_stride == 1 else (att_stride * 2),
                stride=att_stride,
                padding=1 if att_stride == 1 else (att_stride // 2),
                key=key_upsampling,
            )
        else:
            self.out_proj = eqx.nn.Conv2d(
                self.head_dim * self.num_heads,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                key=key_oproj,
            )
            self.upsample = eqx.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3 if att_stride == 1 else (att_stride * 2),
                stride=att_stride,
                padding=1 if att_stride == 1 else (att_stride // 2),
                groups=in_channels,
                key=key_upsampling,
            )

    def __call__(
        self,
        x: Float[Array, "seqlen height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen height width"]:
        q, k, v = rearrange(
            self.qkv(x),
            "(n h d) s1 s2 -> n h (s1 s2) d",
            n=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        _, s, _ = q.shape
        h = w = int(s**0.5)

        attn_logits = q @ rearrange(k, "h s d -> h d s")
        attn_logits /= self.head_dim**0.5

        if self.attention_type == "softmax":
            attn = jax.nn.softmax(attn_logits, axis=-1)
        elif self.attention_type == "sigmoid":
            attn = jax.nn.sigmoid(attn_logits)

        v = attn @ v

        out = self.out_proj(rearrange(v, "h (s1 s2) d -> (h d) s1 s2", s1=h, s2=w))
        out = self.upsample(out)

        return out


class ConvAttentionBlock(eqx.Module):
    prenorm: eqx.Module
    postnorm: eqx.Module
    norm: eqx.Module
    attn: eqx.Module
    mlp: eqx.Module
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        in_channels: int,
        *,
        key: PRNGKeyArray,
        mlp_ratio: float = 4.0,
        att_stride: int = 1,
        attention_type: Literal["softmax", "sigmoid"] = "softmax",
        fuse: bool = True,  # TODO: verify
        drop_path: float | List[float] = 0.0,
        act_layer: Callable = jax.nn.gelu,  # TODO: try hardswish
        norm_layer: eqx.Module = eqx.nn.GroupNorm,
        norm_max_group: int = 32,
        post_attention_norm: bool = False,
        eps: float = 1e-5,
        **kwargs,
    ):
        key_attn, key_mlp = jr.split(key, 2)

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)} ({drop_path})."
                )
            dr1, dr2 = drop_path
            dr1 = float(dr1)
            dr2 = float(dr2)
        else:
            dr1 = dr2 = float(drop_path)

        num_groups = nearest_power_of_2_divisor(in_channels, norm_max_group)
        self.prenorm = norm_layer(num_groups, in_channels, eps=eps)
        self.postnorm = (
            norm_layer(num_groups, in_channels, eps=eps)
            if post_attention_norm
            else eqx.nn.Identity()
        )
        self.norm = norm_layer(num_groups, in_channels, eps=eps)

        self.attn = ConvAttention(
            in_channels=in_channels,
            att_kernel=5 if att_stride > 1 else 3,
            att_stride=att_stride,
            fuse=fuse,
            attention_type=attention_type,
            key=key_attn,
        )

        self.mlp = ConvBlock(
            dim=in_channels,
            hidden_dim=int(in_channels * mlp_ratio),
            kernel_size=1,
            stride=1,
            padding=0,
            drop_path=dr1,
            act_layer=act_layer,
            key=key_mlp,
        )

        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "seqlen dim"]:
        key_attn, key_mlp, key_dr1, key_dr2 = jr.split(key, 4)

        x = self.drop_path1(
            x,
            self.postnorm(
                self.attn(
                    self.prenorm(x),
                    inference=inference,
                    key=key_attn,
                )
            ),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.mlp(
                self.norm(x),
                inference=inference,
                key=key_mlp,
            ),
            inference=inference,
            key=key_dr2,
        )

        return x


class LowFormerBlock(eqx.Module):
    context_module: ConvAttentionBlock
    local_module: MBConv
    drop_path1: DropPathAdd
    drop_path2: DropPathAdd

    def __init__(
        self,
        in_channels: int,
        *,
        key,
        mlp_ratio: float = 4.0,
        att_stride: int = 1,
        attention_type: Literal["softmax", "sigmoid"] = "softmax",
        fuse_conv: bool = True,
        drop_path: float | List[float] = 0.0,
        act_layer: Callable = jax.nn.hard_swish,
        norm_layer: eqx.Module = eqx.nn.GroupNorm,
        expand_ratio: float = 4.0,
        mbconv_norm_layers: tuple = (None, None, eqx.nn.GroupNorm),
        mbconv_act_layers: tuple = (jax.nn.hard_swish, jax.nn.hard_swish, None),
        fuse_mbconv: bool = False,
        **kwargs,
    ):
        key_context, key_local = jr.split(key, 2)

        if isinstance(drop_path, list):
            if len(drop_path) != 2:
                raise AssertionError(
                    f"`drop_path` needs to have 2 elements, got {len(drop_path)} ({drop_path})."
                )
            dr1, dr2 = drop_path
            dr1 = float(dr1)
            dr2 = float(dr2)
        else:
            dr1 = dr2 = float(drop_path)

        self.context_module = ConvAttentionBlock(
            in_channels=in_channels,
            mlp_ratio=mlp_ratio,
            att_stride=att_stride,
            attention_type=attention_type,
            fuse=fuse_conv,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            key=key_context,
        )
        self.local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            norm_layers=mbconv_norm_layers,
            act_layers=mbconv_act_layers,
            use_bias=(True, True, False),
            fuse=fuse_mbconv,
            key=key_local,
        )

        self.drop_path1 = DropPathAdd(dr1)
        self.drop_path2 = DropPathAdd(dr2)

    def __call__(
        self,
        x: Float[Array, "dim height width"],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ):
        key_context, key_local, key_dr1, key_dr2 = jr.split(key, 4)

        x = self.drop_path1(
            x,
            self.context_module(x, inference=inference, key=key_context),
            inference=inference,
            key=key_dr1,
        )
        x = self.drop_path2(
            x,
            self.local_module(x, inference=inference, key=key_local),
            inference=inference,
            key=key_dr2,
        )

        return x


def get_attention(module: str | eqx.Module) -> eqx.Module:
    """Get an `eqx.Module` from its common name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    match module:
        case "attention":
            return Attention
        case "windowedattention":
            return WindowedAttention
        case "shsa":
            return SHSA
        case "linearattention":
            return LinearAttention
        case "mmsa":
            return MMSA
        case "sqa":
            return SQA
        case "linearangularattention":
            return LinearAngularAttention
        case _:
            raise ValueError(f"Got an unknown module string: {module}")


def get_attention_block(module: str | eqx.Module) -> eqx.Module:
    """Get an `eqx.Module` from its common name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    match module:
        case "attentionblock":
            return AttentionBlock
        case "hatblock":
            return HATBlock
        case "mllablock":
            return MllaBlock
        case "partialformerblock":
            return PartialFormerBlock
        case _:
            raise ValueError(f"Got an unknown module string: {module}")
