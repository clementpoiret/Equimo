from typing import Callable, Optional

import equinox as eqx
import jax
import jax.random as jr
from einops import reduce
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.activation import get_act
from equimo.layers.convolution import MBConv, SingleConvBlock
from equimo.models.registry import register_model

MNLayerConfig = tuple[int, int, int, int, bool, str]


@register_model("mobilenetv3")
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
        num_classes: int | None = 1000,
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
            act_layer="hard_swish",
            key=key_conv1,
        )
        self.layers = tuple(
            MBConv(
                in_channels=in_c,
                out_channels=layers_config[i + 1][0]
                if i + 1 < len(layers_config)
                else last_channels,
                mid_channels=mid_c,
                kernel_size=k,
                stride=s,
                se=se,
                act_layer=(get_act(act), get_act(act), None),
                residual=True,
                key=jr.fold_in(key, i),
            )
            for i, (in_c, mid_c, k, s, se, act) in enumerate(layers_config)
        )

        self.dropout = eqx.nn.Dropout(dropout)
        self.classifier = (
            eqx.nn.Linear(last_channels, num_classes, key=key_clf)
            if num_classes is not None and num_classes > 0
            else eqx.nn.Identity()
        )

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


_MOBILENET_BASE_CFG: dict = {
    "in_channels": 3,
}

_MOBILENET_REGISTRY: dict[str, tuple[dict, dict]] = {
    "mobilenetv3_small": (
        _MOBILENET_BASE_CFG,
        {
            "layers_config": [
                (16, 16, 3, 2, True, "relu"),
                (24, 72, 3, 2, False, "relu"),
                (24, 88, 3, 1, False, "relu"),
                (40, 96, 5, 2, True, "hard_swish"),
                (40, 240, 5, 1, True, "hard_swish"),
                (40, 240, 5, 1, True, "hard_swish"),
                (48, 120, 5, 1, True, "hard_swish"),
                (48, 144, 5, 1, True, "hard_swish"),
                (96, 288, 5, 2, True, "hard_swish"),
                (96, 576, 5, 1, True, "hard_swish"),
                (96, 576, 5, 1, True, "hard_swish"),
            ],
        },
    ),
    "mobilenetv3_large": (
        _MOBILENET_BASE_CFG,
        {
            "layers_config": [
                (16, 16, 3, 1, False, "relu"),
                (24, 64, 3, 2, False, "relu"),
                (24, 72, 3, 1, False, "relu"),
                (40, 72, 5, 2, True, "relu"),
                (40, 120, 5, 1, True, "relu"),
                (40, 120, 5, 1, True, "relu"),
                (80, 240, 3, 2, False, "hard_swish"),
                (80, 200, 3, 1, False, "hard_swish"),
                (80, 184, 3, 1, False, "hard_swish"),
                (80, 184, 3, 1, False, "hard_swish"),
                (112, 480, 3, 1, True, "hard_swish"),
                (112, 672, 3, 1, True, "hard_swish"),
                (160, 672, 5, 2, True, "hard_swish"),
                (160, 960, 5, 1, True, "hard_swish"),
                (160, 960, 5, 1, True, "hard_swish"),
            ],
        },
    ),
}


def _build_mobilenet(
    variant: str,
    pretrained: bool = False,
    inference_mode: bool = True,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> MobileNetv3:
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _MOBILENET_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    model = MobileNetv3(**cfg, key=key)

    if pretrained:
        from equimo.io import load_weights

        model = load_weights(
            model,
            identifier=variant,
            inference_mode=inference_mode,
        )

    return model


def mobilenetv3_small(**kwargs) -> MobileNetv3:
    """MobileNetV3-Small — 11 MBConv blocks, last_channels 1280."""
    return _build_mobilenet("mobilenetv3_small", **kwargs)


def mobilenetv3_large(**kwargs) -> MobileNetv3:
    """MobileNetV3-Large — 15 MBConv blocks, last_channels 1280."""
    return _build_mobilenet("mobilenetv3_large", **kwargs)
