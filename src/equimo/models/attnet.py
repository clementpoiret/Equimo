from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.generic import BlockChunk
from equimo.layers.norm import get_norm
from equimo.models.registry import register_model


@register_model("attnet")
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
        act_layer: str | Callable = "gelu",
        norm_layer: str | type[eqx.Module] = "layernorm",
        num_classes: int | None = 1000,
        eps=1e-5,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_blk, key_head = jr.split(key, 2)

        depth = sum(depths)

        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0.0, drop_path_rate, depth).tolist()

        blocks = []
        _bc_dim = [in_channels, *dims[:-1]]
        for i, _k in enumerate(jr.split(key_blk, len(dims))):
            blocks.append(
                BlockChunk(
                    depth=depths[i],
                    in_channels=_bc_dim[i],
                    out_channels=dims[i],
                    module="atconvblock",
                    module_kwargs={
                        "dim": dims[i],
                        "kernel_size": kernel_sizes[i],
                        "exp_rate": exp_rates[i],
                        "act_layer": act_layer,
                        "glu_norm": glu_norm[i],
                        "glu_dwconv": glu_dwconv[i],
                        "use_bias": conv_bias,
                        "dropout": dropout,
                        "use_layer_scale": use_layer_scale,
                    },
                    downsampler="convnormdownsampler",
                    downsampler_kwargs={"mode": "double" if i == 0 else "simple"},
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    key=_k,
                )
            )
        self.blocks = tuple(blocks)

        self.norm = norm_layer(dims[-1], eps=eps)
        self.head = (
            eqx.nn.Linear(dims[-1], num_classes, key=key_head)
            if num_classes is not None and num_classes > 0
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


def attnet_xxs(**kwargs) -> AttNet:
    return AttNet(
        depths=[2, 2, 4, 2],
        dims=[32, 64, 128, 240],
        exp_rates=[8, 8, 4, 4],
        glu_dwconv=[True, True, True, True],
        glu_norm=[False, False, False, False],
        kernel_sizes=[3, 3, 3, 3],
        drop_path_rate=0.02,
        use_layer_scale=False,
        **kwargs,
    )


def attnet_xs(**kwargs) -> AttNet:
    return AttNet(
        depths=[2, 2, 7, 2],
        dims=[40, 80, 160, 320],
        exp_rates=[8, 8, 4, 4],
        glu_dwconv=[True, True, True, True],
        glu_norm=[False, False, False, False],
        kernel_sizes=[3, 3, 3, 3],
        drop_path_rate=0.02,
        use_layer_scale=True,
        **kwargs,
    )


def attnet_s(**kwargs) -> AttNet:
    return AttNet(
        depths=[2, 3, 10, 3],
        dims=[40, 80, 160, 320],
        exp_rates=[8, 8, 4, 4],
        glu_dwconv=[True, True, True, True],
        glu_norm=[False, False, False, False],
        kernel_sizes=[3, 3, 3, 3],
        drop_path_rate=0.1,
        use_layer_scale=False,
        **kwargs,
    )


def attnet_t1(**kwargs) -> AttNet:
    return AttNet(
        depths=[2, 3, 12, 3],
        dims=[48, 96, 224, 384],
        exp_rates=[8, 8, 4, 4],
        glu_dwconv=[True, True, True, True],
        kernel_sizes=[3, 3, 3, 3],
        drop_path_rate=0.1,
        **kwargs,
    )


def attnet_t2(**kwargs) -> AttNet:
    return AttNet(
        depths=[3, 3, 16, 3],
        dims=[64, 128, 288, 512],
        exp_rates=[8, 8, 4, 4],
        glu_dwconv=[True, True, True, True],
        kernel_sizes=[3, 3, 3, 3],
        drop_path_rate=0.2,
        **kwargs,
    )


def attnet_t3(**kwargs) -> AttNet:
    return AttNet(
        depths=[4, 4, 26, 4],
        dims=[72, 144, 320, 576],
        exp_rates=[8, 8, 4, 4],
        glu_dwconv=[True, True, True, True],
        kernel_sizes=[3, 3, 3, 3],
        drop_path_rate=0.4,
        **kwargs,
    )


def attnet_t4(**kwargs) -> AttNet:
    return AttNet(
        depths=[5, 5, 28, 5],
        dims=[96, 192, 384, 768],
        exp_rates=[8, 8, 4, 4],
        glu_dwconv=[True, True, True, True],
        kernel_sizes=[3, 3, 3, 3],
        drop_path_rate=0.5,
        **kwargs,
    )
