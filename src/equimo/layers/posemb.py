import math
from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from equimo.layers.convolution import SingleConvBlock

_POSEMB_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_posemb(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new positional embedding modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.

    Args:
        name: Registry key. Defaults to the lowercase class name.
        force: If True, allow overwriting an existing entry. Default False.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _POSEMB_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_POSEMB_REGISTRY[registry_name]}."
            )

        _POSEMB_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_posemb(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get a positional embedding `eqx.Module` class from its registered name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _POSEMB_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_POSEMB_REGISTRY.keys())}"
        )

    return _POSEMB_REGISTRY[module_lower]


def _rotate_half(x: jax.Array) -> jax.Array:
    x_paired = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_paired[..., 0], x_paired[..., 1]
    return jnp.stack([-x2, x1], axis=-1).reshape(x.shape)


@register_posemb()
class LearnedPosEmbed(eqx.Module):
    weight: jax.Array

    dim: int = eqx.field(static=True)
    embed_size: int = eqx.field(static=True)
    num_prefix_tokens: int = eqx.field(static=True)
    num_embedded_prefix_tokens: int = eqx.field(static=True)
    global_pos_embed_cls: bool = eqx.field(static=True)
    global_pos_embed_reg: bool = eqx.field(static=True)

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
        grid = pe[num_embedded_prefix_tokens:]
        grid = rearrange(grid, "(h w) d -> h w d", h=old_size[0], w=old_size[1])
        grid = jax.image.resize(
            grid, (H, W, dim), method=interpolation, antialias=self.antialias
        )
        grid = rearrange(grid, "h w d -> (h w) d")
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
            if C != self.dim:
                raise ValueError(f"Channel dim mismatch: {C} vs {self.dim}")
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
            if cls_token.shape[-1] != self.dim or cls_token.shape[0] != 1:
                raise ValueError(
                    f"cls_token must have shape (1, {self.dim}), got {cls_token.shape}"
                )
            to_cat.append(cls_token)
        if reg_tokens is not None:
            if reg_tokens.ndim != 2 or reg_tokens.shape[-1] != self.dim:
                raise ValueError(
                    f"reg_tokens must have shape (R, {self.dim}), got {reg_tokens.shape}"
                )
            to_cat.append(reg_tokens)

        if not self.global_pos_embed_cls:
            # Add pos to patches only; then prepend any prefix tokens (cls/reg)
            x = x + pos_embed
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=0)

        elif self.global_pos_embed_reg:
            # All prefix tokens (cls + reg) are included in the positional grid; concat first, then add
            if to_cat:
                x = jnp.concatenate(to_cat + [x], axis=0)
            x = x + pos_embed

        else:
            # Only the class token is embedded with patches; reg tokens (if any) are inserted after
            # the class token and before the patch tokens without global posemb.
            # Note: this branch assumes that if reg_tokens are used, a cls_token exists too.
            if cls_token is None and reg_tokens is not None:
                raise ValueError(
                    "Configuration invalid: reg_tokens without cls_token when "
                    "global_pos_embed_reg=False and global_pos_embed_cls=True."
                )
            x = jnp.concatenate(to_cat[:1] + [x], axis=0)  # cat cls_token if present
            x = x + pos_embed
            if reg_tokens is not None:
                # Insert reg_tokens between cls and patch tokens
                x = jnp.concatenate([x[:1], reg_tokens, x[1:]], axis=0)

        return x


@register_posemb()
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
    relative_coords_table: Float[Array, "..."]

    def __init__(
        self,
        dim: int,
        rank: int,
        seq_len: int,
        *,
        key: PRNGKeyArray,
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
        coords_table = jax.lax.stop_gradient(self.relative_coords_table)
        if self.rank == 1:
            table = coords_table
        else:
            table = rearrange(coords_table, "c h w -> (h w) c")

        pos_emb = jax.vmap(self.cpb_mlp)(table)

        return x + pos_emb.astype(x.dtype)


@register_posemb()
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
    relative_coords_table: Float[Array, "..."]
    relative_position_index: Integer[Array, "..."]

    def __init__(
        self,
        window_size: Tuple[int, int],
        pretrained_window_size: Tuple[int, int],
        num_heads: int,
        seq_len: int,
        *,
        key: PRNGKeyArray,
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
        coords_table = jax.lax.stop_gradient(self.relative_coords_table)
        position_index = jax.lax.stop_gradient(self.relative_position_index)
        relative_position_bias_table = jax.vmap(jax.vmap(jax.vmap(self.cpb_mlp)))(
            coords_table
        ).reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            position_index.reshape(-1)
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


@register_posemb()
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
        # Reshape x to separate real and imaginary parts
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x_r, x_i = x_reshaped[..., 0], x_reshaped[..., 1]

        # Apply rotation via real arithmetic: (x_r + i·x_i)(r_r + i·r_i)
        rotations_ng = jax.lax.stop_gradient(self.rotations)
        r_r = rotations_ng[..., 0].astype(x.dtype)
        r_i = rotations_ng[..., 1].astype(x.dtype)
        pe_r = x_r * r_r - x_i * r_i
        pe_i = x_r * r_i + x_i * r_r

        return jnp.stack([pe_r, pe_i], axis=-1).reshape(*x.shape)


@register_posemb()
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
        key: PRNGKeyArray,
        inference: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute (sin, cos) with shapes [H*W, D_head].

        If `inference is False`, training-time augmentations may be applied
        depending on configuration. If `inference is True`, no augmentations
        are applied.
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


@register_posemb()
class VisionRoPE(eqx.Module):
    """Unified 2-D Vision Rotary Position Embedding.

    Supports two frequency strategies:

    * **period-based** (DinoRoPE style):
        Supply ``base`` *or* ``(min_period, max_period)`` with
        ``num_heads``.  Coordinates are normalised to [-1, 1] via
        ``normalize_coords``, with optional training-time augmentations.

    * **mode-based** (VisionRotaryEmbedding style):
        Supply ``freqs_for`` (one of 'lang', 'pixel', 'constant') with
        ``pt_seq_len``.  Positions are divided by the axis length and
        rescaled to the pretrained grid size.

    In both cases ``get_sincos(H=..., W=...)`` returns ``(sin, cos)`` with
    shape ``(H*W, D_out)`` so downstream code doesn't need to know which
    strategy was used.
    """

    # common
    freqs: Float[Array, "F"]
    dtype: jnp.dtype = eqx.field(static=True)

    # strategy selector
    strategy: Literal["period", "mode"] = eqx.field(static=True)

    # period-based fields (DinoRoPE)
    D_head: Optional[int] = eqx.field(static=True, default=None)
    normalize_coords: Optional[Literal["min", "max", "separate"]] = eqx.field(
        static=True, default=None
    )
    shift_coords: Optional[float] = eqx.field(static=True, default=None)
    jitter_coords: Optional[float] = eqx.field(static=True, default=None)
    rescale_coords: Optional[float] = eqx.field(static=True, default=None)

    # mode-based fields (VisionRotaryEmbedding)
    pt_seq_len: Optional[int] = eqx.field(static=True, default=None)

    def __init__(
        self,
        strategy: Literal["period", "mode"],
        *,
        # period-based params
        dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        periods_dtype: jnp.dtype = jnp.bfloat16,
        # mode-based params
        pt_seq_len: int = 14,
        freqs_for: str = "lang",
        theta: float = 10000,
        max_freq: float = 10,
        num_freqs: int = 1,
        custom_freqs: Optional[jax.Array] = None,
        # common
        dtype: jnp.dtype = jnp.float32,
    ):
        self.strategy = strategy
        self.dtype = dtype

        if strategy == "period":
            if dim is None or num_heads is None:
                raise ValueError("strategy='period' requires `dim` and `num_heads`.")
            if dim % (4 * num_heads) != 0:
                raise ValueError("dim must be divisible by 4 * num_heads.")
            both = (min_period is not None) and (max_period is not None)
            if (base is None and not both) or (base is not None and both):
                raise ValueError(
                    "Exactly one of `base` or `min_period`+`max_period` required."
                )

            D_head = dim // num_heads
            D_quarter = D_head // 4

            if base is not None:
                denom = D_head // 2
                k = jnp.arange(D_quarter, dtype=periods_dtype)
                freqs = base ** (2.0 * k / float(denom))
            else:
                assert min_period is not None and max_period is not None
                ratio = max_period / min_period
                exp = jnp.linspace(0.0, 1.0, D_quarter, dtype=periods_dtype)
                freqs = ratio**exp / ratio * max_period
                freqs = freqs.astype(periods_dtype)

            self.freqs = freqs.astype(dtype)
            self.D_head = D_head
            self.normalize_coords = normalize_coords
            self.shift_coords = shift_coords
            self.jitter_coords = jitter_coords
            self.rescale_coords = rescale_coords
            self.pt_seq_len = None

        elif strategy == "mode":
            # When called from VisionTransformer, dim=embed_dim. For mode strategy
            # we need dim_rope = head_dim // 2 = dim // num_heads // 2
            if num_heads is not None and dim is not None:
                dim = dim // num_heads // 2

            if custom_freqs is not None:
                freqs = jnp.asarray(custom_freqs)
            elif freqs_for == "lang":
                if dim is None:
                    raise ValueError("strategy='mode' requires `dim`.")
                freqs = 1.0 / (
                    theta
                    ** (jnp.arange(0, dim, 2)[: dim // 2].astype(jnp.float32) / dim)
                )
            elif freqs_for == "pixel":
                if dim is None:
                    raise ValueError("strategy='mode' requires `dim`.")
                freqs = jnp.linspace(1.0, max_freq / 2.0, dim // 2) * jnp.pi
            elif freqs_for == "constant":
                freqs = jnp.ones(num_freqs, dtype=jnp.float32)
            else:
                raise ValueError(f"Unknown modality '{freqs_for}'")

            self.freqs = freqs.astype(dtype)
            self.pt_seq_len = pt_seq_len
            self.D_head = None
            self.normalize_coords = None
            self.shift_coords = None
            self.jitter_coords = None
            self.rescale_coords = None

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'period' or 'mode'.")

    def _coords_period(
        self, H: int, W: int, *, key: PRNGKeyArray, inference: bool
    ) -> jax.Array:
        """Normalised coords in [-1, 1] with optional augmentations, shape (H*W, 2)."""
        dtype = self.dtype

        if self.normalize_coords == "max":
            d = float(max(H, W))
            ch = jnp.arange(0.5, H, step=1.0) / d
            cw = jnp.arange(0.5, W, step=1.0) / d
        elif self.normalize_coords == "min":
            d = float(min(H, W))
            ch = jnp.arange(0.5, H, step=1.0) / d
            cw = jnp.arange(0.5, W, step=1.0) / d
        else:
            ch = jnp.arange(0.5, H, step=1.0) / float(H)
            cw = jnp.arange(0.5, W, step=1.0) / float(W)

        hh, ww = jnp.meshgrid(ch, cw, indexing="ij")
        coords = jnp.stack([hh, ww], axis=-1).reshape(H * W, 2)
        coords = (2.0 * coords - 1.0).astype(dtype)

        if inference:
            return coords

        k_shift, k_jitter, k_rescale = jax.random.split(key, 3)

        if self.shift_coords is not None:
            shift = jax.random.uniform(
                k_shift, (2,), minval=-self.shift_coords, maxval=self.shift_coords
            ).astype(dtype)
            coords = coords + shift[None, :]

        if self.jitter_coords is not None:
            if self.jitter_coords <= 0:
                raise ValueError("jitter_coords must be > 0.")
            jm = jnp.log(jnp.asarray(self.jitter_coords, dtype=dtype))
            jitter = jnp.exp(
                jax.random.uniform(k_jitter, (2,), minval=-jm, maxval=jm)
            ).astype(dtype)
            coords = coords * jitter[None, :]

        if self.rescale_coords is not None:
            if self.rescale_coords <= 0:
                raise ValueError("rescale_coords must be > 0.")
            rm = jnp.log(jnp.asarray(self.rescale_coords, dtype=dtype))
            rescale = jnp.exp(
                jax.random.uniform(k_rescale, (1,), minval=-rm, maxval=rm)
            ).astype(dtype)
            coords = coords * rescale

        return coords

    def _coords_mode(self, H: int, W: int) -> Tuple[jax.Array, jax.Array]:
        """Position vectors rescaled by pt_seq_len, shapes (H,) and (W,)."""
        assert self.pt_seq_len is not None
        t_h = jnp.arange(H, dtype=jnp.float32) / H * self.pt_seq_len
        t_w = jnp.arange(W, dtype=jnp.float32) / W * self.pt_seq_len
        return t_h, t_w

    # def get_sincos(
    #     self,
    #     *,
    #     H: int,
    #     W: int,
    #     key: Optional[PRNGKeyArray] = None,
    #     inference: bool = True,
    # ) -> Tuple[jax.Array, jax.Array]:
    #     """Compute ``(sin, cos)`` each with shape ``(H*W, D_out)``.

    #     For period-based: ``D_out = D_head``.
    #     For mode-based:   ``D_out = 2 * dim`` (height + width concatenated).

    #     ``key`` and ``inference`` are only used by the period strategy
    #     (for augmentations) and can be omitted for mode-based usage.
    #     """
    #     dtype = self.dtype
    #     freqs = jax.lax.stop_gradient(self.freqs).astype(dtype)

    #     if self.strategy == "period":
    #         if key is None and not inference:
    #             raise ValueError(
    #                 "A PRNG key is required for period-based RoPE during training."
    #             )
    #         if key is None:
    #             key = jax.random.PRNGKey(0)
    #         D_quarter = self.D_head // 4

    #         coords = self._coords_period(H, W, key=key, inference=inference)
    #         angles = (2.0 * jnp.pi * coords[:, :, None]) / freqs[None, None, :]
    #         angles = angles.reshape(H * W, 2 * D_quarter)
    #         angles = jnp.tile(angles, (1, 2))

    #     else:  # "mode"
    #         t_h, t_w = self._coords_mode(H, W)

    #         freqs_h = jnp.outer(t_h, freqs)
    #         freqs_w = jnp.outer(t_w, freqs)
    #         freqs_h = jnp.repeat(freqs_h, 2, axis=-1)
    #         freqs_w = jnp.repeat(freqs_w, 2, axis=-1)

    #         D = freqs_h.shape[-1]
    #         fh = jnp.broadcast_to(freqs_h[:, None, :], (H, W, D))
    #         fw = jnp.broadcast_to(freqs_w[None, :, :], (H, W, D))
    #         angles = jnp.concatenate([fh, fw], axis=-1).reshape(H * W, -1)

    #     return jnp.sin(angles).astype(dtype), jnp.cos(angles).astype(dtype)
    def get_sincos(
        self,
        *,
        H: int,
        W: int,
        num_prefix_tokens: int = 0,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute ``(sin, cos)`` each with shape ``(num_prefix_tokens + H*W, D_out)``.

        Prefix tokens (CLS, registers, etc.) receive identity rotation
        (sin=0, cos=1) so that RoPE leaves them unchanged.
        """
        dtype = self.dtype
        freqs = jax.lax.stop_gradient(self.freqs).astype(dtype)

        if self.strategy == "period":
            if key is None and not inference:
                raise ValueError(
                    "A PRNG key is required for period-based RoPE during training."
                )
            if key is None:
                key = jax.random.PRNGKey(0)
            D_quarter = self.D_head // 4

            coords = self._coords_period(H, W, key=key, inference=inference)
            angles = (2.0 * jnp.pi * coords[:, :, None]) / freqs[None, None, :]
            angles = angles.reshape(H * W, 2 * D_quarter)
            angles = jnp.tile(angles, (1, 2))

        else:  # "mode"
            t_h, t_w = self._coords_mode(H, W)

            freqs_h = jnp.outer(t_h, freqs)
            freqs_w = jnp.outer(t_w, freqs)
            freqs_h = jnp.repeat(freqs_h, 2, axis=-1)
            freqs_w = jnp.repeat(freqs_w, 2, axis=-1)

            D = freqs_h.shape[-1]
            fh = jnp.broadcast_to(freqs_h[:, None, :], (H, W, D))
            fw = jnp.broadcast_to(freqs_w[None, :, :], (H, W, D))
            angles = jnp.concatenate([fh, fw], axis=-1).reshape(H * W, -1)

        sin = jnp.sin(angles).astype(dtype)
        cos = jnp.cos(angles).astype(dtype)

        # Prepend identity rotation for prefix tokens (CLS, registers, etc.)
        if num_prefix_tokens > 0:
            D_out = sin.shape[-1]
            prefix_sin = jnp.zeros((num_prefix_tokens, D_out), dtype=dtype)
            prefix_cos = jnp.ones((num_prefix_tokens, D_out), dtype=dtype)
            sin = jnp.concatenate([prefix_sin, sin], axis=0)
            cos = jnp.concatenate([prefix_cos, cos], axis=0)

        return sin, cos

    # def __call__(
    #     self,
    #     x: Float[Array, "..."],
    #     *,
    #     key: Optional[PRNGKeyArray] = None,
    #     inference: bool = True,
    # ) -> Float[Array, "..."]:
    #     """Apply rotary embedding. Expects ``x.shape[-3]`` to be ``H * W``."""
    #     seq_len = x.shape[-3]
    #     ft = int(seq_len**0.5)
    #     sin, cos = self.get_sincos(H=ft, W=ft, key=key, inference=inference)
    #     cos = cos[:, None, :]
    #     sin = sin[:, None, :]
    #     return x * cos + _rotate_half(x) * sin
    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        num_prefix_tokens: int = 0,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = True,
    ) -> Float[Array, "..."]:
        """Apply rotary embedding. Expects ``x.shape[-3]`` to be
        ``num_prefix_tokens + H * W`` with a square spatial grid.
        """
        seq_len = x.shape[-3] - num_prefix_tokens
        ft = int(seq_len**0.5)
        sin, cos = self.get_sincos(
            H=ft,
            W=ft,
            num_prefix_tokens=num_prefix_tokens,
            key=key,
            inference=inference,
        )
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        return x * cos + _rotate_half(x) * sin


@register_posemb()
class CompositeVisionRoPE(eqx.Module):
    """Composite RoPE that applies different rotary embeddings to different
    token groups within a single sequence.

    Token layout:

        [ CLS | reg_0 ... reg_{R-1} | patch_0 ... patch_{H*W-1} ]

    - **CLS** (and any other prefix tokens): identity rotation (pass-through).
    - **Register tokens**: separate RoPE on a ``(h, w)`` grid, typically
      with a much higher frequency base so registers occupy a distinct
      region of position space.
    - **Patch tokens**: main RoPE on the ``(H, W)`` spatial grid.

    Parameters
    ----------
    patch_rope : VisionRoPE
        RoPE instance for spatial patch tokens.
    reg_rope : VisionRoPE | None
        RoPE instance for register tokens.  If None, registers receive
        identity rotation.
    num_prefix_tokens : int
        Number of leading tokens to leave unrotated (e.g. 1 for [CLS]).
    num_registers : int
        Number of register tokens (immediately after prefix).
    reg_grid : tuple[int, int] | None
        Explicit (h, w) grid for registers.  If None, inferred as
        ``(int(sqrt(num_registers)), int(sqrt(num_registers)))`` which
        requires ``num_registers`` to be a perfect square.
    """

    patch_rope: VisionRoPE
    reg_rope: Optional[VisionRoPE]
    num_prefix_tokens: int = eqx.field(static=True)
    num_registers: int = eqx.field(static=True)
    reg_grid: Tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        patch_rope: VisionRoPE,
        *,
        reg_rope: Optional[VisionRoPE] = None,
        num_prefix_tokens: int = 1,
        num_registers: int = 0,
        reg_grid: Optional[Tuple[int, int]] = None,
    ):
        self.patch_rope = patch_rope
        self.reg_rope = reg_rope
        self.num_prefix_tokens = num_prefix_tokens
        self.num_registers = num_registers

        if num_registers > 0 and reg_grid is None:
            side = int(num_registers**0.5)
            if side * side != num_registers:
                raise ValueError(
                    f"num_registers={num_registers} is not a perfect square. "
                    f"Provide `reg_grid` explicitly."
                )
            reg_grid = (side, side)
        self.reg_grid = reg_grid if reg_grid is not None else (0, 0)

    def get_sincos(
        self,
        *,
        H: int,
        W: int,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute ``(sin, cos)`` for the full sequence.

        Returns shape ``(num_prefix + num_registers + H*W, D_out)``.
        """
        sin_p, cos_p = self.patch_rope.get_sincos(
            H=H,
            W=W,
            key=key,
            inference=inference,
        )
        D_out = sin_p.shape[-1]
        dtype = sin_p.dtype

        parts_sin = []
        parts_cos = []

        # 1. Prefix: identity
        if self.num_prefix_tokens > 0:
            parts_sin.append(jnp.zeros((self.num_prefix_tokens, D_out), dtype=dtype))
            parts_cos.append(jnp.ones((self.num_prefix_tokens, D_out), dtype=dtype))

        # 2. Registers
        if self.num_registers > 0 and self.reg_rope is not None:
            rh, rw = self.reg_grid
            sin_r, cos_r = self.reg_rope.get_sincos(
                H=rh,
                W=rw,
                key=key,
                inference=inference,
            )
            parts_sin.append(sin_r)
            parts_cos.append(cos_r)
        elif self.num_registers > 0:
            parts_sin.append(jnp.zeros((self.num_registers, D_out), dtype=dtype))
            parts_cos.append(jnp.ones((self.num_registers, D_out), dtype=dtype))

        # 3. Patches
        parts_sin.append(sin_p)
        parts_cos.append(cos_p)

        return jnp.concatenate(parts_sin, axis=0), jnp.concatenate(parts_cos, axis=0)

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        H: Optional[int] = None,
        W: Optional[int] = None,
        key: Optional[PRNGKeyArray] = None,
        inference: bool = True,
    ) -> Float[Array, "..."]:
        """Apply composite RoPE to a full sequence.

        If ``H`` and ``W`` are not given, the patch grid is inferred
        as square from ``seq_len - num_prefix_tokens - num_registers``.
        """
        seq_len = x.shape[-3]
        n_patches = seq_len - self.num_prefix_tokens - self.num_registers

        if H is None or W is None:
            ft = int(n_patches**0.5)
            H, W = ft, ft

        sin, cos = self.get_sincos(H=H, W=W, key=key, inference=inference)
        cos = cos[:, None, :]
        sin = sin[:, None, :]
        return x * cos + _rotate_half(x) * sin


@register_posemb()
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
        h = w = int(math.isqrt(l))
        if h * w != l:
            raise ValueError(
                f"PosCNN requires a square sequence length, got {l} (not a perfect square)."
            )

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


@register_posemb()
class PosCNN2D(eqx.Module):
    """Convolutional Position Encoding for 2D inputs.

    Uses depthwise convolutions to capture local spatial relationships
    in 2D feature maps. Similar to PosCNN but operates directly on
    2D inputs without reshaping.

    Attributes:
        residual: Whether to add a residual connection (static)
        proj: Depthwise convolution block
    """

    residual: bool = eqx.field(static=True)
    proj: SingleConvBlock

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int | None = None,
        stride: int = 1,
        norm_layer: str | type[eqx.Module] | None = "groupnorm",
        residual: bool = True,
        key: PRNGKeyArray,
        **kwargs,
    ):
        effective_out = out_channels or in_channels
        self.residual = residual and (stride == 1 and in_channels == effective_out)

        self.proj = SingleConvBlock(
            in_channels=in_channels,
            out_channels=effective_out,
            groups=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm_layer=norm_layer,
            act_layer=None,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray,
        inference: bool = False,
    ) -> Float[Array, "channels height width"]:
        x1 = self.proj(x, inference=inference, key=key)

        if self.residual:
            return x + x1
        else:
            return x1
