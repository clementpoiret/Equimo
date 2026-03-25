from functools import partial
from typing import Callable

import jax


def get_act(activation: str | Callable) -> Callable:
    """Get an activation function from its common name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(activation, str):
        return activation

    match activation:
        case "relu":
            return jax.nn.relu
        case "gelu":
            return jax.nn.gelu
        case "exactgelu":
            return partial(jax.nn.gelu, approximate=False)
        case "silu":
            return jax.nn.silu
        case "elu":
            return jax.nn.elu
        case "sigmoid":
            return jax.nn.sigmoid
        case "hard_sigmoid":
            return jax.nn.hard_sigmoid
        case "hard_swish":
            return jax.nn.hard_swish
        case "softmax":
            return jax.nn.softmax
        case _:
            raise ValueError(f"Got an unknown activation string: {activation}")
