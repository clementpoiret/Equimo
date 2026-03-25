from typing import Callable, Optional

import equinox as eqx
import jax

from equimo.utils import nearest_power_of_2_divisor

from ._base import (
    AbstractUpdater,
)

_UPDATER_REGISTRY: dict[str, type[AbstractUpdater]] = {}


def register_updater(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[AbstractUpdater]], type[AbstractUpdater]]:
    """Decorator to dynamically register new updater modules.

    Args:
        name: Registry key. Defaults to the lowercase class name.
        force: If True, allow overwriting an existing entry. Default False.
    """

    def decorator(cls: type[AbstractUpdater]) -> type[AbstractUpdater]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _UPDATER_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_UPDATER_REGISTRY[registry_name]}."
            )

        _UPDATER_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_updater(module: str | type[AbstractUpdater]) -> type[AbstractUpdater]:
    """Get an updater `eqx.Module` class from its registered name."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _UPDATER_REGISTRY:
        raise ValueError(
            f"Got an unknown updater string: '{module}'. "
            f"Available updaters: {list(_UPDATER_REGISTRY.keys())}"
        )

    return _UPDATER_REGISTRY[module_lower]


@register_updater(name="identity")
class Identity(AbstractUpdater):
    """
    z_{k+1} = z_proposed

    Use this ONLY if the layer in the block contains a residual connection internally
    (e.g., `z = z + conv(z)`). Otherwise, gradients may vanish or the fixed point
    may be trivial although it may still work for shallow networks.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, z, z_next, x, *, inference=False, key=None) -> jax.Array:
        return z_next


@register_updater(name="projected_output")
class ProjectedOutput(AbstractUpdater):
    """
    z_{k+1} = GroupNorm(z_next)

    Projects the layer output directly onto the normalized manifold.
    Unlike NormProjected (which sums z + z_next), this treats the layer
    as a direct map to the sphere rather than a residual field.
    """

    norm: eqx.nn.GroupNorm

    def __init__(self, dim: int, **kwargs):
        num_groups = nearest_power_of_2_divisor(dim, 32)
        self.norm = eqx.nn.GroupNorm(num_groups, dim)

    def __call__(self, z, z_next, x, *, inference=False, key=None) -> jax.Array:
        return self.norm(z_next)
