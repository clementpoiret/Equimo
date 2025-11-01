from typing import Callable, Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.convolution import ATConvBlock
from equimo.layers.norm import get_norm
from equimo.utils import nearest_power_of_2_divisor


class Downsampler(eqx.Module):
    downsampler: eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        act_layer: Callable = jax.nn.gelu,
        mode: Literal["double", "simple"] = "simple",
        key: PRNGKeyArray,
    ):
        match mode:
            case "simple":
                self.downsampler = eqx.nn.Sequential(
                    [
                        eqx.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            key=key,
                        ),
                        eqx.nn.GroupNorm(
                            nearest_power_of_2_divisor(out_channels, 32), out_channels
                        ),
                    ]
                )
            case "double":
                self.downsampler = eqx.nn.Sequential(
                    [
                        eqx.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=(_d := out_channels // 2),
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            key=jr.fold_in(key, 0),
                        ),
                        eqx.nn.GroupNorm(nearest_power_of_2_divisor(_d, 32), _d),
                        eqx.nn.Lambda(act_layer),
                        eqx.nn.Conv2d(
                            in_channels=_d,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            key=jr.fold_in(key, 1),
                        ),
                        eqx.nn.GroupNorm(
                            nearest_power_of_2_divisor(out_channels, 32), out_channels
                        ),
                    ]
                )

    def __call__(self, x):
        return self.downsampler(x)


class BlockChunk(eqx.Module):
    blocks: Tuple[ATConvBlock, ...]
    downsample: Downsampler

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        downsampler_mode: Literal["simple", "double"] = "simple",
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_ds, *block_subkeys = jr.split(key, depth + 1)
        keys_to_spread = [
            k for k, v in kwargs.items() if isinstance(v, list) and len(v) == depth
        ]
        out_channels = kwargs["in_channels"]

        self.downsample = Downsampler(
            dim, out_channels, mode=downsampler_mode, key=key_ds
        )

        blocks = []
        for i in range(depth):
            config = kwargs | {k: kwargs[k][i] for k in keys_to_spread}
            blocks.append(ATConvBlock(**config, key=block_subkeys[i]))
        self.blocks = tuple(blocks)

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "..."]:
        keys = jr.split(key, len(self.blocks))

        x = self.downsample(x)
        for blk, key_block in zip(self.blocks, keys):
            x = blk(x, inference=inference, key=key_block, **kwargs)

        return x


class AttNet(eqx.Module):
    blocks: Tuple[BlockChunk, ...]
    norm: eqx.Module
    head: eqx.nn.Linear | eqx.nn.Identity

    def __init__(
        self,
        in_channels: int,
        *,
        dims: list[int] = [64, 128, 256, 512],
        depths: list[int] = [2, 2, 6, 2],
        exp_rates: list[int] = [4, 4, 4, 4],
        kernel_sizes: list[int] = [5, 5, 5, 5],
        conv_bias: bool = True,
        use_layer_scale: bool = True,
        glu_dwconv: list[bool] = [True, True, True, True],
        glu_norm: list[bool] = [True, True, True, True],
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        drop_path_uniform: bool = False,
        act_layer: Callable | str = jax.nn.gelu,
        norm_layer: str | eqx.Module = eqx.nn.LayerNorm,
        num_classes: int = 1000,
        eps=1e-5,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_down, key_blk, key_head = jr.split(key, 3)

        depth = sum(depths)

        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = list(jnp.linspace(0.0, drop_path_rate, depth))

        blocks = []
        _bc_dim = [in_channels, *dims[:-1]]
        for i, _k in enumerate(jr.split(key_blk, len(dims))):
            blocks.append(
                BlockChunk(
                    dim=_bc_dim[i],
                    depth=depths[i],
                    downsampler_mode="double" if i == 0 else "simple",
                    in_channels=dims[i],
                    kernel_size=kernel_sizes[i],
                    exp_rate=exp_rates[i],
                    act_layer=act_layer,
                    glu_norm=glu_norm[i],
                    glu_dwconv=glu_dwconv[i],
                    use_bias=conv_bias,
                    dropout=dropout,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    use_layer_scale=True,
                    key=_k,
                )
            )
        self.blocks = tuple(blocks)

        self.norm = norm_layer(dims[-1], eps=eps)
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
    ) -> Float[Array, "num_classes"]:
        key_blocks = jr.split(key, len(self.blocks))
        for blk, key_blk in zip(self.blocks, key_blocks):
            x = blk(x, inference=inference, key=key_blk)
        return x

    def __call__(
        self,
        x: Float[Array, "channels height width"],
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "num_classes"]:
        x = self.features(x, inference=inference, key=key)
        x = self.norm(x.mean((1, 2)))
        x = self.head(x)

        return x
