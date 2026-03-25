from typing import Callable, Optional

import equinox as eqx
import jax

from ._base import AbstractFuser

_FUSER_REGISTRY: dict[str, type[AbstractFuser]] = {}


def register_fuser(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[AbstractFuser]], type[AbstractFuser]]:
    """Decorator to dynamically register new fuser modules.

    Args:
        name: Registry key. Defaults to the lowercase class name.
        force: If True, allow overwriting an existing entry. Default False.
    """

    def decorator(cls: type[AbstractFuser]) -> type[AbstractFuser]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _FUSER_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_FUSER_REGISTRY[registry_name]}."
            )

        _FUSER_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_fuser(module: str | type[AbstractFuser]) -> type[AbstractFuser]:
    """Get a fuser `eqx.Module` class from its registered name."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _FUSER_REGISTRY:
        raise ValueError(
            f"Got an unknown fuser string: '{module}'. "
            f"Available fusers: {list(_FUSER_REGISTRY.keys())}"
        )

    return _FUSER_REGISTRY[module_lower]


@register_fuser(name="add")
class Add(AbstractFuser):
    """z_in = z + x. Requires x and z to have same shape."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, z, x, *, inference=False, key=None) -> jax.Array:
        # x[0] if x is Context
        raw_x = x[0] if isinstance(x, tuple) else x
        return z + raw_x
