import math
from typing import Any, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray


class LearnedPosEmbed(eqx.Module):
    weight: jax.Array

    dim: int = eqx.field(static=True)
    embed_size: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    num_embedded_prefix_tokens: int = eqx.field(static=True)
    no_embed_class: bool = eqx.field(static=True)
    pos_embed_reg_tokens: bool = eqx.field(static=True)

    antialias: bool = eqx.field(static=True, default=True)

    def resample(
        self,
        *,
        new_size: Tuple[int, int],
        dim: int | None = None,
        num_embedded_prefix_tokens: int | None = None,
        old_size: Optional[Tuple[int, int]] = None,
        interpolation: str = "bicubic",
    ) -> jax.Array:
        """Resample positional embeddings for different input sizes.

        Args:
            new_size: Target size (height, width)
            dim: Dimensionality of the sequence
            num_embedded_prefix_tokens: To include cls and reg tokens
            old_size: Original size (height, width), computed if None
            interpolation: Interpolation method

        Returns:
            Resampled positional embeddings
        """
        pe = self.weight
        prev_dtype = pe.dtype
        H, W = new_size
        dim = self.dim if dim is None else dim
        num_embedded_prefix_tokens = (
            self.num_embedded_prefix_tokens
            if num_embedded_prefix_tokens is None
            else num_embedded_prefix_tokens
        )

        tgt_len = H * W + num_embedded_prefix_tokens
        if (
            (tgt_len == pe.shape[0])
            and (old_size is not None)
            and (H == W == old_size[0])
        ):
            return pe

        if old_size is None:
            L = pe.shape[0] - num_embedded_prefix_tokens
            hw = int(math.sqrt(L))
            old_size = (hw, hw)

        prefix = pe[:num_embedded_prefix_tokens] if num_embedded_prefix_tokens else None
        grid = pe[num_embedded_prefix_tokens:].astype(jnp.float32)
        grid = rearrange(grid, "(h w) d -> h w d", h=old_size[0], w=old_size[1])
        grid = jax.image.resize(
            grid, (H, W, dim), method=interpolation, antialias=self.antialias
        )
        grid = rearrange(grid, "h w d -> (h w) d").astype(prev_dtype)
        if prefix is not None:
            grid = jnp.concatenate([prefix, grid], axis=0)
        return grid

    def __call__(
        self,
        x: jax.Array,
        *,
        cls_token: Optional[jax.Array],
        reg_tokens: Optional[jax.Array],
        dynamic_img_size: bool,
        interpolation: str = "bicubic",
    ) -> jax.Array:
        """Compose tokens and add positional embeddings.

        Inputs:
        - x:
          - If dynamic_img_size: shape (C, H, W) from PatchEmbedding(flatten=False)
          - Else: shape ((H*W), C) from PatchEmbedding(flatten=True)
        - cls_token: shape (1, dim) or None
        - reg_tokens: shape (R, dim) or None
        - dynamic_img_size: whether x is spatial or already flattened

        Returns:
        - Token sequence with positional information and optional prefix tokens.
        """
        if dynamic_img_size:
            C, H, W = x.shape
            assert C == self.dim, f"Channel dim mismatch: {C} vs {self.dim}"
            pos_embed = self.resample(
                new_size=(H, W),
                old_size=(self.embed_size, self.embed_size),
                interpolation=interpolation,
            )
            x = rearrange(x, "c h w -> (h w) c")
        else:
            pos_embed = self.weight

        to_cat = []
        if cls_token is not None:
            # Expect (1, dim)
            assert cls_token.shape[-1] == self.dim and cls_token.shape[0] == 1
            to_cat.append(cls_token)
        if reg_tokens is not None:
            # Expect (R, dim)
            assert reg_tokens.ndim == 2 and reg_tokens.shape[-1] == self.dim
            to_cat.append(reg_tokens)

        # Branching exactly mirrors your current _pos_embed logic
        if self.no_embed_class:
            # Add pos to patches only; then prepend any prefix tokens (cls/reg)
            x = x + pos_embed
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=0)

        elif self.pos_embed_reg_tokens:
            # Prefix tokens are included in the positional grid length; concat first, then add
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=0)
            x = x + pos_embed

        else:
            # Only class token is embedded with patches; reg tokens (if any) are inserted after
            # the class token and before the patch tokens.
            # Note: this branch assumes that if reg_tokens are used, a cls_token exists too.
            if cls_token is None and reg_tokens is not None:
                raise ValueError(
                    "Configuration invalid: reg_tokens without cls_token when pos_embed_reg_tokens=False "
                    "and no_embed_class=False."
                )
            x = jnp.concatenate(to_cat[:1] + [x], axis=0)  # cat cls_token if present
            x = x + pos_embed
            if reg_tokens is not None:
                # Insert reg_tokens between cls and patch tokens
                x = jnp.concatenate([x[:1], reg_tokens, x[1:]], axis=0)

        return x


class PosEmbMLPSwinv1D(eqx.Module):
    """1D Positional Embedding using MLP for Swin Transformer.

    Implements learnable relative position embeddings using an MLP network.
    Supports both 1D sequences and 2D images flattened to 1D.

    Attributes:
        rank: Dimensionality of position encoding (1 for 1D, 2 for 2D)
        seq_len: Length of input sequence
        cpb_mlp: MLP network for computing position embeddings
        relative_coords_table: Table of relative coordinates (static)
    """

    rank: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)

    cpb_mlp: eqx.Module
    relative_coords_table: jnp.ndarray = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        rank: int,
        seq_len: int,
        *,
        key=PRNGKeyArray,
        **kwargs,
    ):
        key1, key2 = jr.split(key, 2)
        self.rank = rank
        self.seq_len = seq_len

        self.cpb_mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(
                    in_features=self.rank,
                    out_features=512,
                    key=key1,
                ),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(
                    in_features=512,
                    out_features=dim,
                    use_bias=False,
                    key=key2,
                ),
            ]
        )

        if self.rank == 1:
            relative_coords_h = jnp.arange(0, seq_len)
            relative_coords_h -= seq_len // 2
            relative_coords_h /= seq_len // 2
            self.relative_coords_table = relative_coords_h[:, jnp.newaxis]
        else:
            seq_len = int(seq_len**0.5)
            relative_coords_h = jnp.arange(0, seq_len)
            relative_coords_w = jnp.arange(0, seq_len)
            relative_coords_table = jnp.stack(
                jnp.meshgrid(relative_coords_h, relative_coords_w)
            )
            relative_coords_table -= seq_len // 2
            relative_coords_table /= seq_len // 2
            self.relative_coords_table = relative_coords_table

    def __call__(
        self,
        x: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        if self.rank == 1:
            table = self.relative_coords_table
        else:
            table = rearrange(self.relative_coords_table, "c h w -> (h w) c")

        pos_emb = jax.vmap(self.cpb_mlp)(table)

        return x + pos_emb.astype(x.dtype)


class PosEmbMLPSwinv2D(eqx.Module):
    """2D Positional Embedding using MLP for Swin Transformer V2.

    Implements learnable relative position embeddings for 2D windows with
    support for cross-window connections and pretrained model adaptation.

    Attributes:
        ct_correct: Whether to use cross-window token correction
        num_heads: Number of attention heads
        seq_len: Length of input sequence
        window_size: Size of local attention window
        cpb_mlp: MLP for computing position bias
        relative_coords_table: Table of relative coordinates (static)
        relative_position_index: Index mapping for relative positions (static)
    """

    ct_correct: bool = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)
    window_size: Tuple[int, int] = eqx.field(static=True)

    cpb_mlp: eqx.nn.Sequential
    relative_coords_table: jnp.ndarray = eqx.field(static=True)
    relative_position_index: jnp.ndarray = eqx.field(static=True)

    def __init__(
        self,
        window_size: Tuple[int, int],
        pretrained_window_size: Tuple[int, int],
        num_heads: int,
        seq_len: int,
        *,
        key=PRNGKeyArray,
        inference: bool = False,
        no_log: bool = False,
        ct_correct: bool = False,
        **kwargs,
    ):
        key1, key2 = jr.split(key, 2)

        self.window_size = window_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.ct_correct = ct_correct

        self.cpb_mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(2, 512, use_bias=True, key=key1),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(512, num_heads, use_bias=False, key=key2),
            ]
        )

        relative_coords_h = jnp.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=jnp.float32
        )
        relative_coords_w = jnp.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=jnp.float32
        )
        relative_coords_table = jnp.stack(
            jnp.meshgrid(relative_coords_h, relative_coords_w)
        )
        relative_coords_table = rearrange(
            relative_coords_table,
            "c h w -> 1 h w c",
        )

        if pretrained_window_size[0] > 0:
            relative_coords_table = relative_coords_table.at[:, :, :, 0].set(
                relative_coords_table[:, :, :, 0] / pretrained_window_size[0] - 1
            )
            relative_coords_table = relative_coords_table.at[:, :, :, 1].set(
                relative_coords_table[:, :, :, 1] / pretrained_window_size[1] - 1
            )
        else:
            relative_coords_table = relative_coords_table.at[:, :, :, 0].set(
                relative_coords_table[:, :, :, 0] / window_size[0] - 1
            )
            relative_coords_table = relative_coords_table.at[:, :, :, 1].set(
                relative_coords_table[:, :, :, 1] / window_size[1] - 1
            )

        if not no_log:
            relative_coords_table = relative_coords_table * 8
            relative_coords_table = (
                jnp.sign(relative_coords_table)
                * jnp.log2(jnp.abs(relative_coords_table) + 1.0)
                / jnp.log2(8)
            )

        self.relative_coords_table = relative_coords_table

        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(1, 2, 0)
        relative_coords = relative_coords + jnp.array(
            [self.window_size[0] - 1, self.window_size[1] - 1]
        )
        relative_coords = relative_coords.at[:, :, 0].set(
            relative_coords[:, :, 0] * (2 * self.window_size[1] - 1)
        )
        self.relative_position_index = jnp.sum(relative_coords, axis=-1)

    def __call__(
        self, x: Float[Array, "..."], local_window_size: int
    ) -> Float[Array, "..."]:
        relative_position_bias_table = jax.vmap(jax.vmap(jax.vmap(self.cpb_mlp)))(
            self.relative_coords_table
        ).reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))
        relative_position_bias = 16 * jax.nn.sigmoid(relative_position_bias)

        n_global_feature = x.shape[2] - local_window_size
        if n_global_feature > 0 and self.ct_correct:
            step_for_ct = self.window_size[0] / (n_global_feature**0.5 + 1)
            seq_len = int(n_global_feature**0.5)
            indices = []

            # TODO: REMOVE THIS FOR LOOPS
            for i in range(seq_len):
                for j in range(seq_len):
                    ind = (i + 1) * step_for_ct * self.window_size[0] + (
                        j + 1
                    ) * step_for_ct
                    indices.append(int(ind))

            top_part = relative_position_bias[:, indices, :]
            lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
            left_part = relative_position_bias[:, :, indices]

        relative_position_bias = jnp.pad(
            relative_position_bias,
            ((0, 0), (n_global_feature, 0), (n_global_feature, 0)),
        )

        if n_global_feature > 0 and self.ct_correct:
            relative_position_bias = relative_position_bias * 0.0
            relative_position_bias = relative_position_bias.at[
                :, :n_global_feature, :n_global_feature
            ].set(lefttop_part)
            relative_position_bias = relative_position_bias.at[
                :, :n_global_feature, n_global_feature:
            ].set(top_part)
            relative_position_bias = relative_position_bias.at[
                :, n_global_feature:, :n_global_feature
            ].set(left_part)

        return x + relative_position_bias.astype(x.dtype)


class RoPE(eqx.Module):
    """Rotary Position Embedding (RoPE).

    Implements rotary position embeddings that encode positions through
    rotation in complex space. This allows the model to naturally capture
    relative positions through rotational differences.

    Attributes:
        rotations: Precomputed rotation matrices for position encoding
    """

    rotations: eqx.Module

    def __init__(self, shape: tuple, base: int = 10000):
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        if feature_dim % k_max != 0:
            raise ValueError("`feature_dim` is not divisible by `k_max`.")

        # angles
        theta_ks = jnp.power(base, -jnp.arange(k_max) / k_max)
        angles = jnp.concatenate(
            [
                t[..., None] * theta_ks
                for t in jnp.meshgrid(
                    *[jnp.arange(d) for d in channel_dims], indexing="ij"
                )
            ],
            axis=-1,
        )

        # rotations
        rotations_re = jnp.cos(angles)
        rotations_im = jnp.sin(angles)
        self.rotations = jnp.stack([rotations_re, rotations_im], axis=-1)

    def __call__(
        self,
        x: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        dtype = x.dtype
        x = x.astype(jnp.float32)

        # Reshape x to separate real and imaginary parts
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = x_reshaped[..., 0] + 1j * x_reshaped[..., 1]

        # Apply rotation
        rotations_ng = jax.lax.stop_gradient(self.rotations)
        rotations_complex = rotations_ng[..., 0] + 1j * rotations_ng[..., 1]
        pe_x = rotations_complex * x_complex

        # Convert back to real representation
        pe_x_real = jnp.stack([pe_x.real, pe_x.imag], axis=-1)

        return pe_x_real.reshape(*x.shape).astype(dtype)


class DinoRoPE(eqx.Module):
    """Axial RoPE that produces per-position sin/cos for later rotation of features.

    - Enforces dim % (4 * num_heads) == 0.
    - Periods can be specified via `base` or `min_period` + `max_period` (mutually exclusive).
    - Coordinates are normalized to [-1, 1] according to `normalize_coords`.
    - Optional training-time augmentations: shift, jitter (log-uniform per-axis), rescale (log-uniform shared).
    - Returns (sin, cos) with shape [H*W, D_head], where D_head = dim // num_heads.

    Parameters
    ----------
    dim: int
        Total embedding dimension (across heads).
    num_heads: int
        Number of attention heads.
    base: float | None
        Period base. Mutually exclusive with (min_period, max_period).
    min_period, max_period: float | None
        Range for geometric periods. Mutually exclusive with base.
    normalize_coords: {"min", "max", "separate"}
        Normalization scheme mapping pixel centers to [-1, 1].
    shift_coords: float | None
        If set and training, add uniform shift in [-shift_coords, +shift_coords] per axis.
    jitter_coords: float | None
        If set and training, multiply each axis by log-uniform in [1/jitter_coords, jitter_coords].
    rescale_coords: float | None
        If set and training, multiply both axes by a shared log-uniform in [1/rescale_coords, rescale_coords].
    dtype: jnp.dtype | None
        Computation/output dtype. Defaults to float32.

    Notes
    -----
    - The `periods` buffer is persistent (part of the tree) and not trainable; we
      stop gradients on it inside `__call__`.
    - I had to separate `dtype` and `periods_dtype`. For some obscure reasons, I faced cases
      with the reference PyTorch impl. where `periods` were computed in bfloat16 (wanted behavior),
      but subsequent computations (coords, angles, cos, sin) were at a float32 precision.
    """

    D_head: int = eqx.field(static=True)
    normalize_coords: Literal["min", "max", "separate"] = eqx.field(static=True)
    shift_coords: Optional[float] = eqx.field(static=True)
    jitter_coords: Optional[float] = eqx.field(static=True)
    rescale_coords: Optional[float] = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)

    # Persistent, non-trainable buffer
    periods: Float[Array, "..."]

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        periods_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.float32,
    ):
        if dim % (4 * num_heads) != 0:
            raise ValueError("dim must be divisible by 4 * num_heads.")
        both_periods = (min_period is not None) and (max_period is not None)
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided (mutually exclusive)."
            )
        if normalize_coords not in ("min", "max", "separate"):
            raise ValueError(f"Unknown normalize_coords: {normalize_coords}")

        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        self.D_head = dim // num_heads
        D_quarter = self.D_head // 4

        if base is not None:
            denom = self.D_head // 2
            k = jnp.arange(D_quarter, dtype=periods_dtype)
            periods = base ** (2.0 * k / float(denom))
        else:
            # Geometric progression from min_period to max_period (inclusive endpoints behavior per torch linspace)
            assert min_period is not None and max_period is not None
            base_ratio = max_period / min_period
            exponents = jnp.linspace(0.0, 1.0, D_quarter, dtype=periods_dtype)
            periods = base_ratio**exponents  # in [1, base_ratio]
            periods = periods / base_ratio  # in [1/base_ratio, 1]
            periods = periods * max_period  # in [min_period, max_period]
            periods = periods.astype(periods_dtype)

        # Persistent buffer (will be copied with the tree; we stop gradients in __call__)
        self.periods = periods.astype(dtype)

    def _make_coords(self, H: int, W: int) -> jnp.ndarray:
        """Create normalized coords in [-1, 1], shape [H*W, 2], dtype=self.dtype."""
        dtype = self.dtype
        # WARNING: I removed `dtype=dtype` in those jnp.arange fns because it was
        # creating a discrepancy w/ dinov3 pytorch impl.

        if self.normalize_coords == "max":
            denom = float(max(H, W))
            coords_h = jnp.arange(0.5, H, step=1.0) / denom  # [H]
            coords_w = jnp.arange(0.5, W, step=1.0) / denom  # [W]
        elif self.normalize_coords == "min":
            denom = float(min(H, W))
            coords_h = jnp.arange(0.5, H, step=1.0) / denom
            coords_w = jnp.arange(0.5, W, step=1.0) / denom
        else:  # "separate"
            coords_h = jnp.arange(0.5, H, step=1.0) / float(H)
            coords_w = jnp.arange(0.5, W, step=1.0) / float(W)

        hh, ww = jnp.meshgrid(coords_h, coords_w, indexing="ij")  # [H, W]
        coords = jnp.stack([hh, ww], axis=-1).reshape(H * W, 2)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # [0,1] -> [-1,1]

        return coords.astype(dtype)

    def get_sincos(
        self,
        *,
        H: int,
        W: int,
        key: jax.Array,
        inference: Optional[bool] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute (sin, cos) with shapes [H*W, D_head].

        If `inference is False`, training-time augmentations may be applied
        depending on configuration. If `inference is True`, no augmentations
        are applied. If `inference is None`, defaults to training mode
        (augmentations applied when configured).
        """
        k_shift, k_jitter, k_rescale = jax.random.split(key, 3)

        dtype = self.dtype
        D_head = self.D_head
        D_quarter = D_head // 4

        coords = self._make_coords(H, W)  # [HW, 2]

        # Shift
        if not inference and (self.shift_coords is not None):
            shift_hw = jax.random.uniform(
                k_shift, shape=(2,), minval=-self.shift_coords, maxval=self.shift_coords
            ).astype(dtype)
            coords = coords + shift_hw[None, :]

        # Jitter (log-uniform per-axis)
        if not inference and (self.jitter_coords is not None):
            if self.jitter_coords <= 0:
                raise ValueError("jitter_coords must be > 0.")
            jitter_max = jnp.log(jnp.asarray(self.jitter_coords, dtype=dtype))
            jitter_min = -jitter_max
            jitter_hw = jax.random.uniform(
                k_jitter, shape=(2,), minval=jitter_min, maxval=jitter_max
            )
            jitter_hw = jnp.exp(jitter_hw).astype(dtype)  # in [1/jitter, jitter]
            coords = coords * jitter_hw[None, :]

        # Rescale (log-uniform shared across both axes)
        if not inference and (self.rescale_coords is not None):
            if self.rescale_coords <= 0:
                raise ValueError("rescale_coords must be > 0.")
            rescale_max = jnp.log(jnp.asarray(self.rescale_coords, dtype=dtype))
            rescale_min = -rescale_max
            rescale = jax.random.uniform(
                k_rescale, shape=(1,), minval=rescale_min, maxval=rescale_max
            )
            rescale = jnp.exp(rescale).astype(dtype)  # in [1/rescale, rescale]
            coords = coords * rescale  # broadcast to both axes

        # Angles
        # angles: [HW, 2, D_quarter] where periods: [D_quarter]
        periods = jax.lax.stop_gradient(self.periods).astype(dtype)
        angles = (2.0 * jnp.pi * coords[:, :, None]) / periods[
            None, None, :
        ]  # [HW, 2, D_quarter]
        angles = angles.reshape(angles.shape[0], 2 * D_quarter)  # [HW, D_head//2]
        angles = jnp.tile(angles, reps=(1, 2))  # [HW, D_head]

        cos = jnp.cos(angles).astype(dtype)  # [HW, D_head]
        sin = jnp.sin(angles).astype(dtype)  # [HW, D_head]

        return sin, cos


class PosCNN(eqx.Module):
    """Convolutional Position Encoding for 1D sequences.

    Uses depthwise convolutions to capture local spatial relationships
    and generate position-aware representations. Input is reshaped from
    1D sequence to 2D for convolution operations.

    Attributes:
        s: Stride for convolution operation (static)
        proj: Depthwise convolution layer
    """

    s: int = eqx.field(static=True)
    proj: eqx.nn.Conv

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        s: int = 1,
        **kwargs,
    ):
        self.proj = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            groups=out_channels,
            kernel_size=3,
            stride=s,
            padding=1,
            key=key,
        )

        self.s = s

    def __call__(
        self,
        x: Float[Array, "seqlen dim"],
    ) -> Float[Array, "seqlen dim"]:
        l, _ = x.shape
        h = w = int(l**0.5)

        x1 = rearrange(
            self.proj(
                rearrange(
                    x,
                    "(h w) c -> c h w",
                    h=h,
                    w=w,
                )
            ),
            "c h w -> (h w) c",
        )

        if self.s == 1:
            return x + x1
        else:
            return x1


class PosCNN2D(eqx.Module):
    """Convolutional Position Encoding for 2D inputs.

    Uses depthwise convolutions to capture local spatial relationships
    in 2D feature maps. Similar to PosCNN but operates directly on
    2D inputs without reshaping.

    Attributes:
        s: Stride for convolution operation (static)
        proj: Depthwise convolution layer
    """

    s: int = eqx.field(static=True)
    proj: eqx.nn.Conv

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        s: int = 1,
        **kwargs,
    ):
        self.proj = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            groups=out_channels,
            kernel_size=3,
            stride=s,
            padding=1,
            key=key,
        )

        self.s = s

    def __call__(
        self,
        x: Float[Array, "channels height width"],
    ) -> Float[Array, "channels height width"]:
        x1 = self.proj(x)

        if self.s == 1:
            return x + x1
        else:
            return x1
