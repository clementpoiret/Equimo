from functools import partial
from typing import Callable, Optional

import jax

_ACT_REGISTRY: dict[str, Callable] = {}


def register_act(
    name: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """Decorator to dynamically register new activation functions.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core activations, which can silently change model behavior.
    """

    def decorator(fn: Callable) -> Callable:
        registry_name = name.lower() if name else fn.__name__.lower()

        if registry_name in _ACT_REGISTRY:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_ACT_REGISTRY[registry_name]}."
            )

        _ACT_REGISTRY[registry_name] = fn
        return fn

    return decorator


def get_act(activation: str | Callable) -> Callable:
    """Get an activation function from its registered name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(activation, str):
        return activation

    activation_lower = activation.lower()
    if activation_lower not in _ACT_REGISTRY:
        raise ValueError(
            f"Got an unknown activation string: '{activation}'. "
            f"Available activations: {list(_ACT_REGISTRY.keys())}"
        )

    return _ACT_REGISTRY[activation_lower]


# Register built-in JAX activations — these cannot be decorated, so we insert directly.
# Collision checking is skipped here; these are canonical names owned by this module.
_ACT_REGISTRY["relu"] = jax.nn.relu
_ACT_REGISTRY["gelu"] = jax.nn.gelu
_ACT_REGISTRY["exactgelu"] = partial(jax.nn.gelu, approximate=False)
_ACT_REGISTRY["silu"] = jax.nn.silu
_ACT_REGISTRY["elu"] = jax.nn.elu
_ACT_REGISTRY["sigmoid"] = jax.nn.sigmoid
_ACT_REGISTRY["hard_sigmoid"] = jax.nn.hard_sigmoid
_ACT_REGISTRY["hard_swish"] = jax.nn.hard_swish
_ACT_REGISTRY["softmax"] = jax.nn.softmax
