from typing import Callable

import equinox as eqx

_LAYER_REGISTRY: dict[str, type[eqx.Module]] = {}


def _registry_name(cls: type[eqx.Module], name: str | None) -> str:
    return name.lower() if name else cls.__name__.lower()


def _register_module(
    registry: dict[str, type[eqx.Module]],
    cls: type[eqx.Module],
    registry_name: str,
    force: bool,
    registry_label: str,
    *,
    add_to_layer_registry: bool = False,
) -> type[eqx.Module]:
    if not issubclass(cls, eqx.Module):
        raise TypeError(
            f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
        )

    if registry_name in registry and not force:
        raise ValueError(
            f"Cannot register '{registry_name}'. It is already registered "
            f"to {registry[registry_name]}."
        )

    if (
        add_to_layer_registry
        and registry_name in _LAYER_REGISTRY
        and not force
    ):
        raise ValueError(
            f"Cannot register '{registry_name}'. It is already registered "
            f"to {_LAYER_REGISTRY[registry_name]}."
        )

    registry[registry_name] = cls
    if add_to_layer_registry:
        _LAYER_REGISTRY[registry_name] = cls
    return cls


def _resolve_from_registry(
    module: str | type[eqx.Module],
    registry: dict[str, type[eqx.Module]],
    registry_label: str,
) -> type[eqx.Module]:
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in registry:
        raise ValueError(
            f"Got an unknown {registry_label} string: '{module}'. "
            f"Available modules: {list(registry.keys())}"
        )

    return registry[module_lower]


def register_layer(
    name: str | None = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register a tabular layer class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        registry_name = _registry_name(cls, name)
        return _register_module(
            _LAYER_REGISTRY,
            cls,
            registry_name,
            force,
            "tabular layer",
        )

    return decorator


def get_layer(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a tabular layer class from its registered name."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower in _LAYER_REGISTRY:
        return _LAYER_REGISTRY[module_lower]

    from equimo.core.layers import get_layer as get_core_layer

    try:
        return get_core_layer(module)
    except ValueError as error:
        raise ValueError(
            f"Got an unknown tabular layer string: '{module}'. "
            f"Available modules: {list(_LAYER_REGISTRY.keys())}"
        ) from error
