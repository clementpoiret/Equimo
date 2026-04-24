"""DEQ model (ConvNeXt variant).

Thin wrapper around :class:`~.implicit.DEQBlock` embedded in a standard
four-stage ConvNet layout. The FPI stage is configured via three orthogonal
components — injector, stabilizer, strategy — each resolvable from its
registry name or passed directly as an instance / class.

Defaults are minimal at the library level (``add`` + ``identity`` + ``entry``).
Model presets such as :func:`deq_convnext_t` override these with combinations
tuned for the specific backbone. For deep ConvNeXt / iFormer stacks the
recommended override is:

* ``fpi_injector = "prenorm_add"`` — pre-normalize ``z`` and add a 1x1
  projection of ``x`` (precomputed once per forward).
* ``fpi_stabilizer = "projected"`` — ``GroupNorm(z_out)`` to keep iterates
  bounded (MDEQ-style).
* ``fpi_strategy = "entry"`` — inject only at the stack entry; blocks keep
  their own internal residual and normalization.
"""

__all__ = ["DEQ", "deq_convnext_t"]

from typing import Callable, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.implicit import (
    DEQBlock,
    get_injector,
    get_stabilizer,
    get_strategy,
)
from equimo.layers import get_layer
from equimo.layers.activation import get_act
from equimo.layers.norm import get_norm
from equimo.models.registry import register_model
from equimo.utils import make_drop_path_schedule


class BlockChunk(eqx.Module):
    """One model stage: optional downsampler + either a plain block list or a DEQ block."""

    block_type: str = eqx.field(static=True)
    downsample_last: bool = eqx.field(static=True)

    blocks: Tuple[eqx.Module, ...] | None
    z0_block: eqx.Module | None
    deq_block: DEQBlock | None
    downsample: eqx.Module | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        *,
        block_type: str = "normal",
        module: type[eqx.Module] | None = None,
        module_kwargs: dict = {},
        downsampler: type[eqx.Module] | None = None,
        downsampler_kwargs: dict = {},
        z0_module: type[eqx.Module] | None = None,
        z0_module_kwargs: dict = {},
        downsample_last: bool = False,
        # Minimal defaults. Model presets override these for specific backbones.
        injector: str = "add",
        stabilizer: str = "identity",
        strategy: str = "entry",
        tol: float = 1e-3,
        max_steps: int = 50,
        drop_path: float | Sequence[float] = 0.0,
        key: PRNGKeyArray,
    ):
        assert module is not None or downsampler is not None, (
            "Either `module` or `downsampler` must be defined."
        )

        key_ds, key_fpi, *block_subkeys = jr.split(key, depth + 2)
        keys_to_spread = [
            k
            for k, v in module_kwargs.items()
            if isinstance(v, list) and len(v) == depth
        ]

        self.block_type = block_type
        self.downsample_last = downsample_last

        # Channel bookkeeping for main modules and downsampler.
        down_in = in_channels
        down_out = out_channels
        if downsampler is not None:
            block_in = block_out = in_channels if downsample_last else out_channels
        else:
            if depth > 1 and out_channels != in_channels:
                raise ValueError(
                    "Please use a dedicated downsampler to have a block of depth>1."
                )
            if block_type == "fpi" and out_channels != in_channels:
                raise ValueError(
                    "Please use a dedicated downsampler to use FPI blocks."
                )
            block_in = in_channels
            block_out = out_channels

        if module is not None:
            if block_type == "normal":
                blocks = []
                for i in range(depth):
                    config = module_kwargs | {
                        k: module_kwargs[k][i] for k in keys_to_spread
                    }
                    blocks.append(
                        module(
                            in_channels=block_in,
                            out_channels=block_out,
                            drop_path=drop_path[i]
                            if isinstance(drop_path, (list, tuple, jax.Array))
                            else drop_path,
                            **config,
                            key=block_subkeys[i],
                        )
                    )
                self.blocks = tuple(blocks)
                self.z0_block = None
                self.deq_block = None

            elif block_type == "fpi":
                key_z0, key_inj, key_stab, key_strat = jr.split(key_fpi, 4)

                self.blocks = None
                self.z0_block = (
                    z0_module(
                        in_channels=block_in,
                        out_channels=block_out,
                        **z0_module_kwargs,
                        key=key_z0,
                    )
                    if z0_module is not None
                    else None
                )

                injector_cls = get_injector(injector)
                stabilizer_cls = get_stabilizer(stabilizer)
                strategy_cls = get_strategy(strategy)

                injector_obj = injector_cls(dim=block_in, key=key_inj)
                stabilizer_obj = stabilizer_cls(dim=block_in, key=key_stab)
                strategy_obj = strategy_cls(dim=block_in, key=key_strat)

                self.deq_block = DEQBlock(
                    channels=block_in,
                    depth=depth,
                    module=module,
                    module_kwargs={
                        "in_channels": block_in,
                        "out_channels": block_out,
                    }
                    | module_kwargs,
                    injector=injector_obj,
                    stabilizer=stabilizer_obj,
                    strategy=strategy_obj,
                    tol=tol,
                    max_steps=max_steps,
                    key=block_subkeys[0],
                )
            else:
                raise ValueError(f"Unknown block type {block_type}.")
        else:
            # Only downsampler.
            self.blocks = None
            self.z0_block = None
            self.deq_block = None

        self.downsample = (
            downsampler(
                in_channels=down_in,
                out_channels=down_out,
                **downsampler_kwargs,
                key=key_ds,
            )
            if downsampler is not None
            else None
        )

    @property
    def n_blocks(self) -> int:
        n = 2
        if self.deq_block is not None:
            n += 1
        if self.blocks is not None:
            n += len(self.blocks)
        return n

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        z0: Float[Array, "..."] | None = None,
        key: PRNGKeyArray,
        inference: bool = False,
        **kwargs,
    ) -> Tuple[Float[Array, "..."], list]:
        key_down, key_z0, *keys = jr.split(key, self.n_blocks)

        auxs = []

        if not self.downsample_last and self.downsample:
            x = self.downsample(x, inference=inference, key=key_down)

        if self.block_type == "normal" and self.blocks is not None:
            for blk, key_block in zip(self.blocks, keys):
                x = blk(x, inference=inference, key=key_block, **kwargs)
        elif self.block_type == "fpi" and self.deq_block is not None:
            z0 = (
                self.z0_block(x, inference=inference, key=key_z0)
                if z0 is None and self.z0_block is not None
                else z0
            )

            x, aux = self.deq_block(x, z0, inference=inference, key=keys[0])
            auxs.append(aux)

        if self.downsample_last and self.downsample:
            x = self.downsample(x, inference=inference, key=key_down)

        return x, auxs


@register_model("deq")
class DEQ(eqx.Module):
    """Four-stage model with one or more stages promoted to a DEQ block.

    Library-level FPI defaults are minimal: ``add`` injector, ``identity``
    stabilizer, ``entry`` strategy. Override via the ``fpi_*`` kwargs, or use
    a model preset (e.g. :func:`deq_convnext_t`) which ships a tuned combo.
    """

    fpi_index: list[int] = eqx.field(static=True)

    blocks: Tuple[BlockChunk, ...]
    dropout: eqx.nn.Dropout
    norm: eqx.Module
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int,
        *,
        modules: list[str | None] = [None, None, None, None],
        module_kwargs: list[dict] = [{}, {}, {}, {}],
        downsamplers: list[str | None] = [None, None, None, None],
        downsampler_kwargs: list[dict] = [{}, {}, {}, {}],
        downsample_last: bool = False,
        z0_module: str | None = None,
        z0_module_kwargs: dict = {},
        block_types: list[str] = ["normal", "normal", "normal", "normal"],
        dims: list[int] = [64, 128, 256, 512],
        depths: list[int] = [2, 2, 6, 2],
        # --- FPI configuration ---
        # Minimal defaults. Presets override for specific backbones.
        fpi_injector: str = "add",
        fpi_stabilizer: str = "identity",
        fpi_strategy: str = "entry",
        fpi_maxsteps: int = 50,
        fpi_tol: float = 1e-3,
        # -------------------------
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        drop_path_uniform: bool = False,
        act_layer: Callable | str = jax.nn.gelu,
        norm_layer: str | type[eqx.Module] = eqx.nn.LayerNorm,
        num_classes: int = 1000,
        eps: float = 1e-5,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_blk, key_head = jr.split(key, 2)

        act_layer = get_act(act_layer)
        norm_layer_cls = get_norm(norm_layer)
        modules_classes = [get_layer(b) if b is not None else None for b in modules]
        downsamplers_classes = [
            get_layer(b) if b is not None else None for b in downsamplers
        ]
        z0_module_cls = get_layer(z0_module) if z0_module is not None else None

        universal_kwargs = {"act_layer": act_layer}

        dpr = make_drop_path_schedule(drop_path_rate, depths, uniform=drop_path_uniform)

        self.fpi_index = [i for i, t in enumerate(block_types) if t == "fpi"]

        blocks = []
        _bc_dim = [in_channels, *dims[:-1]]
        block_keys = jr.split(key_blk, len(dims))
        for i, _k in enumerate(block_keys):
            blocks.append(
                BlockChunk(
                    in_channels=_bc_dim[i],
                    out_channels=dims[i],
                    depth=depths[i],
                    module=modules_classes[i],
                    module_kwargs=universal_kwargs | module_kwargs[i],
                    downsampler=downsamplers_classes[i],
                    downsampler_kwargs=universal_kwargs | downsampler_kwargs[i],
                    downsample_last=downsample_last,
                    block_type=block_types[i],
                    z0_module=z0_module_cls,
                    z0_module_kwargs=z0_module_kwargs,
                    injector=fpi_injector,
                    stabilizer=fpi_stabilizer,
                    strategy=fpi_strategy,
                    tol=fpi_tol,
                    max_steps=fpi_maxsteps,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    key=_k,
                )
            )
        self.blocks = tuple(blocks)

        self.dropout = eqx.nn.Dropout(p=dropout)
        self.norm = norm_layer_cls(dims[-1], eps=eps)
        self.head = (
            eqx.nn.Linear(dims[-1], num_classes, key=key_head)
            if num_classes > 0
            else eqx.nn.Identity()
        )

    def features(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[Float[Array, "num_classes"], list]:
        key_drop, *key_blocks = jr.split(key, len(self.blocks) + 1)
        auxs = []
        for blk, key_blk in zip(self.blocks, key_blocks):
            x, aux = blk(x, inference=inference, key=key_blk)
            if aux:
                auxs.extend(aux)
        x = self.dropout(x, inference=inference, key=key_drop)
        return x, auxs

    def readout(
        self,
        z_star: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
    ):
        if len(self.fpi_index) == 0:
            raise ValueError("`readout` called but no fpi block is present.")
        if len(self.fpi_index) > 1:
            raise NotImplementedError(
                "`readout` called but with multiple fpi blocks. "
                "This behavior is currently not supported."
            )

        x = z_star
        fpi_index: int = self.fpi_index[0]
        for blk in self.blocks[fpi_index + 1 :]:
            x, _ = blk(x, inference=True, key=key)
        x = self.norm(x.mean((1, 2)))
        x = self.head(x)

        return x

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[Float[Array, "num_classes"], list]:
        x, auxs = self.features(x, inference=inference, key=key)
        x = self.norm(x.mean((1, 2)))
        x = self.head(x)

        return x, auxs


# Model presets.
# The preset ships a tuned FPI combo (pre-norm injection, norm-projection
# stabilization, single entry injection) — the combination that reliably
# converges for a 9-block ConvNeXt FPI stage with LayerScale(1e-6) init.
# Users who call the bare DEQ class get the minimal defaults instead.
_DEQ_BASE_CFG: dict = {
    "in_channels": 3,
    "num_classes": 1000,
    "dims": [96, 192, 384, 768],
    "depths": [3, 3, 9, 3],
    "block_types": ["normal", "normal", "fpi", "normal"],
    "modules": ["convnextblock", "convnextblock", "convnextblock", "convnextblock"],
    "module_kwargs": [{"dim": 96}, {"dim": 192}, {"dim": 384}, {"dim": 768}],
    "downsamplers": [
        "convnextstem",
        "convnextdownsampler",
        "convnextdownsampler",
        "convnextdownsampler",
    ],
    # Tuned FPI combo for a deep ConvNeXt stack.
    "fpi_injector": "prenorm_add",
    "fpi_stabilizer": "projected",
    "fpi_strategy": "entry",
}

_DEQ_REGISTRY: dict[str, tuple[dict, dict]] = {
    "deq_convnext_t": (
        _DEQ_BASE_CFG,
        {},
    ),
}


def _build_deq(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> DEQ:
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _DEQ_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    model = DEQ(**cfg, key=key)

    if pretrained:
        from equimo.io import load_weights

        model = load_weights(
            model,
            identifier=variant,
            inference_mode=inference_mode,
        )

    return model


def deq_convnext_t(**kwargs) -> DEQ:
    return _build_deq("deq_convnext_t", **kwargs)
