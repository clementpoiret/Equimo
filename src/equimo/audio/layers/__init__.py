from typing import Callable, Optional

import equinox as eqx

_AUDIO_LAYER_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_layer(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register an audio-specific layer class."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )
        registry_name = name.lower() if name else cls.__name__.lower()
        if registry_name in _AUDIO_LAYER_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_AUDIO_LAYER_REGISTRY[registry_name]}."
            )
        _AUDIO_LAYER_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_layer(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve an audio-specific layer class."""
    if not isinstance(module, str):
        return module
    module_lower = module.lower()
    if module_lower not in _AUDIO_LAYER_REGISTRY:
        raise ValueError(
            f"Got an unknown audio layer string: '{module}'. "
            f"Available layers: {list(_AUDIO_LAYER_REGISTRY.keys())}"
        )
    return _AUDIO_LAYER_REGISTRY[module_lower]


__all__ = ["get_layer", "register_layer"]
