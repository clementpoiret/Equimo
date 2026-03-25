__all__ = [
    "AttNet",
    "attnet_xxs",
    "attnet_xs",
    "attnet_s",
    "attnet_t1",
    "attnet_t2",
    "attnet_t3",
    "attnet_t4",
]

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


_ATTNET_BASE_CFG: dict = {
    "in_channels": 3,
    "exp_rates": [8, 8, 4, 4],
    "glu_dwconv": [True, True, True, True],
    "kernel_sizes": [3, 3, 3, 3],
}

_ATTNET_REGISTRY: dict[str, tuple[dict, dict]] = {
    "attnet_xxs": (
        _ATTNET_BASE_CFG,
        {
            "depths": [2, 2, 4, 2],
            "dims": [32, 64, 128, 240],
            "glu_norm": [False, False, False, False],
            "drop_path_rate": 0.02,
            "use_layer_scale": False,
        },
    ),
    "attnet_xs": (
        _ATTNET_BASE_CFG,
        {
            "depths": [2, 2, 7, 2],
            "dims": [40, 80, 160, 320],
            "glu_norm": [False, False, False, False],
            "drop_path_rate": 0.02,
            "use_layer_scale": True,
        },
    ),
    "attnet_s": (
        _ATTNET_BASE_CFG,
        {
            "depths": [2, 3, 10, 3],
            "dims": [40, 80, 160, 320],
            "glu_norm": [False, False, False, False],
            "drop_path_rate": 0.1,
            "use_layer_scale": False,
        },
    ),
    "attnet_t1": (
        _ATTNET_BASE_CFG,
        {
            "depths": [2, 3, 12, 3],
            "dims": [48, 96, 224, 384],
            "drop_path_rate": 0.1,
        },
    ),
    "attnet_t2": (
        _ATTNET_BASE_CFG,
        {
            "depths": [3, 3, 16, 3],
            "dims": [64, 128, 288, 512],
            "drop_path_rate": 0.2,
        },
    ),
    "attnet_t3": (
        _ATTNET_BASE_CFG,
        {
            "depths": [4, 4, 26, 4],
            "dims": [72, 144, 320, 576],
            "drop_path_rate": 0.4,
        },
    ),
    "attnet_t4": (
        _ATTNET_BASE_CFG,
        {
            "depths": [5, 5, 28, 5],
            "dims": [96, 192, 384, 768],
            "drop_path_rate": 0.5,
        },
    ),
}


def _build_attnet(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> AttNet:
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _ATTNET_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    model = AttNet(**cfg, key=key)

    if pretrained:
        from equimo.io import load_weights

        model = load_weights(
            model,
            identifier=variant,
            inference_mode=inference_mode,
        )

    return model


def attnet_xxs(**kwargs) -> AttNet:
    """AttNet-XXS — 32→240 dims, depths [2,2,4,2], drop_path 0.02."""
    return _build_attnet("attnet_xxs", **kwargs)


def attnet_xs(**kwargs) -> AttNet:
    """AttNet-XS — 40→320 dims, depths [2,2,7,2], drop_path 0.02."""
    return _build_attnet("attnet_xs", **kwargs)


def attnet_s(**kwargs) -> AttNet:
    """AttNet-S — 40→320 dims, depths [2,3,10,3], drop_path 0.1."""
    return _build_attnet("attnet_s", **kwargs)


def attnet_t1(**kwargs) -> AttNet:
    """AttNet-T1 — 48→384 dims, depths [2,3,12,3], drop_path 0.1."""
    return _build_attnet("attnet_t1", **kwargs)


def attnet_t2(**kwargs) -> AttNet:
    """AttNet-T2 — 64→512 dims, depths [3,3,16,3], drop_path 0.2."""
    return _build_attnet("attnet_t2", **kwargs)


def attnet_t3(**kwargs) -> AttNet:
    """AttNet-T3 — 72→576 dims, depths [4,4,26,4], drop_path 0.4."""
    return _build_attnet("attnet_t3", **kwargs)


def attnet_t4(**kwargs) -> AttNet:
    """AttNet-T4 — 96→768 dims, depths [5,5,28,5], drop_path 0.5."""
    return _build_attnet("attnet_t4", **kwargs)
