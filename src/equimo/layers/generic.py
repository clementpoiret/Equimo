from typing import Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.dropout import DropPathAdd
from equimo.layers.norm import LayerScale


def _resolve_layer(name_or_cls: "str | type[eqx.Module]") -> "type[eqx.Module]":
    """Resolve a layer class from its registered name.

    Searches all known registries in priority order. Accepts a class directly
    (returned as-is) or a string key (looked up across all registries).

    Priority order: attn_block → conv → mixer → posemb → downsampler →
    patch → attn → norm → ffn → dropout → se → wavelet.

    Args:
        name_or_cls: A registered name (case-insensitive) or an eqx.Module subclass.

    Returns:
        The resolved module class.

    Raises:
        ValueError: If the name is not found in any registry.
    """
    if not isinstance(name_or_cls, str):
        return name_or_cls

    name_lower = name_or_cls.lower()

    # Lazy imports to avoid circular dependencies at module load time.
    from equimo.layers.attention import _ATTN_BLOCK_REGISTRY, _ATTN_REGISTRY
    from equimo.layers.convolution import _CONV_REGISTRY
    from equimo.layers.downsample import _DOWNSAMPLER_REGISTRY
    from equimo.layers.dropout import _DROPOUT_REGISTRY
    from equimo.layers.ffn import _FFN_REGISTRY
    from equimo.layers.mamba import _MIXER_REGISTRY
    from equimo.layers.norm import _NORM_REGISTRY
    from equimo.layers.patch import _PATCH_REGISTRY
    from equimo.layers.posemb import _POSEMB_REGISTRY
    from equimo.layers.squeeze_excite import _SE_REGISTRY
    from equimo.layers.wavelet import _WAVELET_REGISTRY

    registries = [
        ("attn_block", _ATTN_BLOCK_REGISTRY),
        ("conv", _CONV_REGISTRY),
        ("mixer", _MIXER_REGISTRY),
        ("posemb", _POSEMB_REGISTRY),
        ("downsampler", _DOWNSAMPLER_REGISTRY),
        ("patch", _PATCH_REGISTRY),
        ("attn", _ATTN_REGISTRY),
        ("norm", _NORM_REGISTRY),
        ("ffn", _FFN_REGISTRY),
        ("dropout", _DROPOUT_REGISTRY),
        ("se", _SE_REGISTRY),
        ("wavelet", _WAVELET_REGISTRY),
    ]

    for _registry_name, registry in registries:
        if name_lower in registry:
            return registry[name_lower]

    available = sorted(
        set().union(*[r.keys() for _, r in registries])
    )
    raise ValueError(
        f"Layer '{name_or_cls}' not found in any registry. "
        f"Available: {available}"
    )


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
        use_ls = all(v is not None for v in [init_values, dim, axis])
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
        block_type: str | type[eqx.Module],
        block_kwargs: dict,
        *,
        window_size: int = 16,
        drop_path: float | list[float] = 0.0,
        key: PRNGKeyArray,
    ):
        self.window_size = window_size
        block_type = _resolve_layer(block_type)
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

        num_windows = x_windows.shape[0]
        window_keys = jr.split(key, num_windows)

        def serial_blocks(x_win, k_seq):
            ks = jr.split(k_seq, len(self.blocks))
            for block, k_blk in zip(self.blocks, ks):
                x_win = block(x_win, key=k_blk, inference=inference)
            return x_win

        x_windows = jax.vmap(serial_blocks)(x_windows, window_keys)

        x_out = rearrange(
            x_windows,
            "(nh nw) c ws_h ws_w -> c (nh ws_h) (nw ws_w)",
            nh=H_pad // self.window_size,
            nw=W_pad // self.window_size,
        )

        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :H, :W]

        return x_out


class BlockChunk(eqx.Module):
    """Universal block chunk for building staged vision architectures.

    Groups a sequence of identical blocks with optional positional embedding
    and downsampling. Supports both pre- and post-downsampling, stochastic
    depth scheduling, and dropout-aware downsampler handling.

    This is the canonical building block for multi-stage models in equimo.
    Blocks and downsamplers are specified by type and receive their kwargs
    through dedicated dicts, keeping the constructor explicit and composable.

    Conventions:
    - All blocks receive ``dim`` as their primary dimension (via ``module_kwargs``).
    - Downsamplers using the channel convention (CNN-style) receive
      ``in_channels`` / ``out_channels`` injected from the top-level params.
    - Downsamplers using the token convention (transformer-style, e.g.
      ``PatchMerging``) should have ``dim`` set in ``downsampler_kwargs``.
    - Any list-valued entry in ``module_kwargs`` whose length equals ``depth``
      is automatically spread across blocks (e.g. per-block attention types).

    Args:
        depth: Number of blocks.
        in_channels: Input channel count. When provided, injected into
            ``downsampler_kwargs`` (via ``setdefault``) as ``in_channels``.
        out_channels: Output channel count. Injected similarly.
        module: Block type to instantiate ``depth`` times. Either ``module``
            or ``downsampler`` must be given.
        module_kwargs: Constructor kwargs shared by all blocks. List-valued
            entries of length ``depth`` are spread per block.
        posemb: Optional positional embedding type applied before blocks.
        posemb_kwargs: Constructor kwargs for the positional embedding.
        downsampler: Optional downsampler type. Applied before or after blocks
            depending on ``downsample_last``.
        downsampler_kwargs: Constructor kwargs for the downsampler.
        downsampler_needs_key: Set ``True`` when the downsampler's ``__call__``
            requires ``key`` and ``inference`` to be forwarded (e.g. it
            contains dropout or other stochastic operations).
        downsample_last: Apply the downsampler after blocks instead of before.
        drop_path: Global or per-block drop-path rate. A single float is
            broadcast; a sequence must have length ``depth``.
        init_values: Layer-scale initialisation value passed to each block.
            Skipped (not forwarded) when ``None``.
        key: PRNG key for parameter initialisation.

    Attributes:
        downsample_last: Static flag for downsampling order.
        downsampler_needs_key: Static flag — ``True`` when the downsampler needs ``key``/``inference``.
        posemb: Positional embedding module (``Identity`` when unused).
        blocks: Tuple of processing blocks, or ``None`` when ``depth == 0``.
        downsample: Downsampling module, or ``None`` when unused.
    """

    downsample_last: bool = eqx.field(static=True)
    downsampler_needs_key: bool = eqx.field(static=True)

    posemb: eqx.Module
    blocks: Tuple[eqx.Module, ...] | None
    downsample: eqx.Module | None

    def __init__(
        self,
        depth: int,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        module: str | type[eqx.Module] | None = None,
        module_kwargs: dict = {},
        posemb: str | type[eqx.Module] | None = None,
        posemb_kwargs: dict = {},
        downsampler: str | type[eqx.Module] | None = None,
        downsampler_kwargs: dict = {},
        downsampler_needs_key: bool = False,
        downsample_last: bool = False,
        drop_path: float | Sequence[float] = 0.0,
        init_values: float | None = None,
        key: PRNGKeyArray,
    ):
        assert module is not None or downsampler is not None, (
            "At least one of `module` or `downsampler` must be specified."
        )

        # Resolve string names to classes via the layer registries.
        if module is not None:
            module = _resolve_layer(module)
        if posemb is not None:
            posemb = _resolve_layer(posemb)
        if downsampler is not None:
            downsampler = _resolve_layer(downsampler)

        key_ds, key_pos, *block_subkeys = jr.split(key, depth + 2)

        self.downsample_last = downsample_last
        self.downsampler_needs_key = downsampler_needs_key

        if isinstance(drop_path, (int, float)):
            dpr = [float(drop_path)] * depth
        else:
            dpr = list(drop_path)
            if len(dpr) != depth:
                raise ValueError(
                    f"drop_path length {len(dpr)} does not match depth {depth}."
                )

        self.posemb = (
            posemb(**posemb_kwargs, key=key_pos)
            if posemb is not None
            else eqx.nn.Identity()
        )

        if module is not None and depth > 0:
            keys_to_spread = [
                k
                for k, v in module_kwargs.items()
                if isinstance(v, list) and len(v) == depth
            ]
            blocks = []
            for i in range(depth):
                config = module_kwargs | {
                    k: module_kwargs[k][i] for k in keys_to_spread
                }
                block_init_kwargs = {}
                if init_values is not None:
                    block_init_kwargs["init_values"] = init_values
                blocks.append(
                    module(
                        drop_path=dpr[i],
                        **block_init_kwargs,
                        **config,
                        key=block_subkeys[i],
                    )
                )
            self.blocks = tuple(blocks)
        else:
            self.blocks = None

        if downsampler is not None:
            ds_kwargs = downsampler_kwargs.copy()
            if in_channels is not None:
                ds_kwargs.setdefault("in_channels", in_channels)
            if out_channels is not None:
                ds_kwargs.setdefault("out_channels", out_channels)
            self.downsample = downsampler(**ds_kwargs, key=key_ds)
        else:
            self.downsample = None

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        key: PRNGKeyArray,
        inference: bool = False,
        **kwargs,
    ) -> Float[Array, "..."]:
        n_blocks = len(self.blocks) if self.blocks is not None else 0
        key_down, *keys = jr.split(key, n_blocks + 2)

        x = self.posemb(x)

        if not self.downsample_last and self.downsample is not None:
            if self.downsampler_needs_key:
                x = self.downsample(x, inference=inference, key=key_down)
            else:
                x = self.downsample(x)

        if self.blocks is not None:
            for blk, key_block in zip(self.blocks, keys):
                x = blk(x, inference=inference, key=key_block, **kwargs)

        if self.downsample_last and self.downsample is not None:
            if self.downsampler_needs_key:
                x = self.downsample(x, inference=inference, key=key_down)
            else:
                x = self.downsample(x)

        return x
