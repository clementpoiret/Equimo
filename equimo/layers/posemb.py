from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray


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
        rotations_complex = self.rotations[..., 0] + 1j * self.rotations[..., 1]
        pe_x = rotations_complex * x_complex

        # Convert back to real representation
        pe_x_real = jnp.stack([pe_x.real, pe_x.imag], axis=-1)

        return pe_x_real.reshape(*x.shape).astype(dtype)


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
