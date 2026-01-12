from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.random as jr
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.convolution import MBConv, SingleConvBlock

MNAct = Literal["re", "hs"]
MNLayerConfig: type = tuple[int, int, int, int, bool, MNAct]


def get_act(act: MNAct) -> Callable:
    match act:
        case "re":
            return jax.nn.relu
        case "hs":
            return jax.nn.hard_swish
        case _:
            raise ValueError(
                f"Unknown activation, got `{act}`, expected one of [`re`, `hs`]"
            )


class MobileNetv3(eqx.Module):
    conv1: SingleConvBlock
    layers: tuple[MBConv, ...]
    dropout: eqx.nn.Dropout
    classifier: eqx.nn.Linear

    def __init__(
        self,
        in_channels: int,
        layers_config: list[MNLayerConfig],
        *,
        last_channels: int = 1280,
        num_classes: int = 1000,
        dropout: float = 0.0,
        key: PRNGKeyArray,
        **kwargs,
    ):
        key_conv1, key_layers, key_clf = jr.split(key, 3)

        self.conv1 = SingleConvBlock(
            in_channels,
            layers_config[0][0],
            kernel_size=3,
            stride=2,
            act_layer=jax.nn.hard_swish,
            key=key_conv1,
        )
        self.layers = tuple(
            MBConv(
                in_channels=in_c,
                out_channels=layers_config[i + 1][0]
                if i + 1 < len(layers_config)
                else last_channels,
                kernel_size=k,
                stride=s,
                se=se,
                act_layer=(get_act(act), get_act(act), None),
                residual=True,
                key=jr.fold_in(key, i),
            )
            for i, (in_c, mid_c, k, s, se, act) in enumerate(
                MNLayerConfig(layers_config)
            )
        )

        self.dropout = eqx.nn.Dropout(dropout)
        self.classifier = eqx.nn.Linear(last_channels, num_classes, key=key_clf)

    def features(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        key_conv1, key_layers = jr.split(key, 2)

        x = self.conv1(x, inference=inference, key=key_conv1)
        for i, layer in enumerate(self.layers):
            x = layer(x, inference=inference, key=jr.fold_in(key_layers, i))

        return x

    def __call__(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        key_f, key_d = jr.split(key, 2)

        x = self.features(x, inference=inference, key=key_f)
        x = reduce(x, "c h w -> c", "mean")
        x = self.dropout(x, inference=inference, key=key_d)
        x = self.classifier(x)

        return x


def mobilenetv3_small(**kwargs) -> MobileNetv3:
    backbone = MobileNetv3(
        layers_config=[
            (16, 16, 3, 2, True, "re"),
            (24, 72, 3, 2, False, "re"),
            (24, 88, 3, 1, False, "re"),
            (40, 96, 5, 2, True, "hs"),
            (40, 240, 5, 1, True, "hs"),
            (40, 240, 5, 1, True, "hs"),
            (48, 120, 5, 1, True, "hs"),
            (48, 144, 5, 1, True, "hs"),
            (96, 288, 5, 2, True, "hs"),
            (96, 576, 5, 1, True, "hs"),
            (96, 576, 5, 1, True, "hs"),
        ],
        **kwargs,
    )
    return backbone


def mobilenetv3_large(**kwargs) -> MobileNetv3:
    backbone = MobileNetv3(
        layers_config=[
            (16, 16, 3, 1, False, "re"),
            (24, 64, 3, 2, False, "re"),
            (24, 72, 3, 1, False, "re"),
            (40, 72, 5, 2, True, "re"),
            (40, 120, 5, 1, True, "re"),
            (40, 120, 5, 1, True, "re"),
            (80, 240, 3, 2, False, "hs"),
            (80, 200, 3, 1, False, "hs"),
            (80, 184, 3, 1, False, "hs"),
            (80, 184, 3, 1, False, "hs"),
            (112, 480, 3, 1, True, "hs"),
            (112, 672, 3, 1, True, "hs"),
            (160, 672, 5, 2, True, "hs"),
            (160, 960, 5, 1, True, "hs"),
            (160, 960, 5, 1, True, "hs"),
        ],
        **kwargs,
    )
    return backbone
