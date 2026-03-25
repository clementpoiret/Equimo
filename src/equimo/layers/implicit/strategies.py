from typing import Callable, Optional

import equinox as eqx
import jax.random as jr

from ._base import AbstractLayerApply

_STRATEGY_REGISTRY: dict[str, type[AbstractLayerApply]] = {}


def register_strategy(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[AbstractLayerApply]], type[AbstractLayerApply]]:
    """Decorator to dynamically register new layer application strategies.

    Args:
        name: Registry key. Defaults to the lowercase class name.
        force: If True, allow overwriting an existing entry. Default False.
    """

    def decorator(cls: type[AbstractLayerApply]) -> type[AbstractLayerApply]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _STRATEGY_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_STRATEGY_REGISTRY[registry_name]}."
            )

        _STRATEGY_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_strategy(module: str | type[AbstractLayerApply]) -> type[AbstractLayerApply]:
    """Get a strategy `eqx.Module` class from its registered name."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Got an unknown strategy string: '{module}'. "
            f"Available strategies: {list(_STRATEGY_REGISTRY.keys())}"
        )

    return _STRATEGY_REGISTRY[module_lower]


@register_strategy(name="standard")
class StandardLayerApply(AbstractLayerApply):
    """
    Standard DEQ Step:
    1. Fuse: z_in = Fuser(z, x)
    2. Layer: z_out = Layer(z_in)

    Note: The global update (mixing z and z_out) happens outside this strategy.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, layer, z, x, fuser, updater, *, inference, key):
        k_fuse, k_layer = jr.split(key)
        z_in = fuser(z, x, inference=inference, key=k_fuse)
        return layer(z_in, inference=inference, key=k_layer)
