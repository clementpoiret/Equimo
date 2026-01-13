from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.dropout import DropPathAdd
from equimo.layers.norm import LayerScale


class Residual(eqx.Module):
    """A wrapper module that adds a residual connection with optional drop path.

    This module wraps any other module and adds a residual (skip) connection around it.
    It also includes drop path regularization which stochastically drops the residual
    path during training. The computation flow is:
    input -> [main branch: module] + [residual branch: identity with drop path] -> output

    Attributes:
        module: The module to wrap with a residual connection
        drop_path: DropPath module for residual connection regularization
    """

    module: eqx.Module
    ls: LayerScale | eqx.nn.Identity
    drop_path: DropPathAdd

    def __init__(
        self,
        module: eqx.Module,
        dim: int | None = None,
        axis: int | None = None,
        init_values: float | None = None,
        drop_path: float = 0.0,
    ):
        """Initialize the Residual wrapper.

        Args:
            module: The module to wrap with a residual connection
            drop_path: Drop path rate (probability of dropping the residual connection)
                      (default: 0)
        """
        self.module = module
        use_ls = all([init_values, dim, axis])
        self.ls = (
            LayerScale(dim, axis=axis, init_values=init_values)
            if use_ls
            else eqx.nn.Identity()
        )
        self.drop_path = DropPathAdd(drop_path)

    def __call__(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        pass_args: bool = False,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        """Forward pass of the residual block.

        Args:
            x: Input tensor of any shape
            enable_dropout: Whether to enable dropout during training
            key: PRNG key for randomness
            pass_args: Whether to pass enable_dropout and key to the wrapped module
                      (default: False)

        Returns:
            Output tensor with same shape as input, combining the module output
            with the residual connection through drop path
        """
        if pass_args:
            x2 = self.module(x, inference=inference, key=key)
        else:
            x2 = self.module(x)

        return self.drop_path(
            x,
            self.ls(x2),
            inference=inference,
            key=key,
        )


class WindowedSequence(eqx.Module):
    blocks: Tuple[eqx.Module, ...]
    window_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        depth: int,
        block_type: type[eqx.Module],
        block_kwargs: dict,
        *,
        window_size: int = 16,
        drop_path: float | list[float] = 0.0,
        key: PRNGKeyArray,
    ):
        self.window_size = window_size
        keys = jr.split(key, depth)

        if isinstance(drop_path, list):
            dpr = drop_path
            if len(dpr) != depth:
                raise ValueError(f"Got {len(dpr)} values for a depth of {depth}.")
        else:
            dpr = [drop_path] * depth

        internal_kwargs = block_kwargs.copy()
        internal_kwargs["window_size"] = 0

        self.blocks = tuple(
            block_type(
                in_channels=in_channels, drop_path=dpr[i], key=k, **internal_kwargs
            )
            for i, k in enumerate(keys)
        )

    def __call__(
        self, x: Float[Array, "C H W"], key: PRNGKeyArray, inference: bool = False
    ) -> Float[Array, "C H W"]:
        C, H, W = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        x_padded = x
        if pad_h > 0 or pad_w > 0:
            x_padded = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)))

        H_pad, W_pad = x_padded.shape[1], x_padded.shape[2]

        # Reshape to (Num_Windows, C, Win_Size, Win_Size)
        # We treat Num_Windows as a batch dimension for the internal blocks
        x_windows = rearrange(
            x_padded,
            "c (nh ws_h) (nw ws_w) -> (nh nw) c ws_h ws_w",
            ws_h=self.window_size,
            ws_w=self.window_size,
        )

        keys = jr.split(key, len(self.blocks))

        def serial_blocks(x_win, k_seq):
            ks = jr.split(k_seq, len(self.blocks))
            for block, k_blk in zip(self.blocks, ks):
                x_win = block(x_win, key=k_blk, inference=inference)
            return x_win

        x_windows = jax.vmap(serial_blocks)(x_windows, keys[0])

        x_out = rearrange(
            x_windows,
            "(nh nw) c ws_h ws_w -> c (nh ws_h) (nw ws_w)",
            nh=H_pad // self.window_size,
            nw=W_pad // self.window_size,
        )

        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :H, :W]

        return x_out
