import re
from typing import Callable, Optional

import equinox as eqx

# Identifier must only contain alphanumerics, hyphens, and underscores.
# This prevents path traversal when identifiers are embedded in local file paths.
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
_MODEL_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_model(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to register a model class under a serialisable string key.

    Registered names are used by :func:`equimo.io.load_model` and
    :func:`get_model_cls` to reconstruct the model architecture from a saved
    config. Collision checking prevents silent overwrites of core models.

    Args:
        name: Registry key. Defaults to the lowercase class name.
        force: If True, allow overwriting an existing entry. Default False.

    Example::

        @register_model("mynet")
        class MyNet(eqx.Module):
            ...

        model = load_model("mynet", path=Path("mynet.tar.lz4"))
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )
        registry_name = name.lower() if name else cls.__name__.lower()
        if registry_name in _MODEL_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_MODEL_REGISTRY[registry_name]}."
            )
        _MODEL_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_model_cls(cls: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a model class from its registered string key or pass it through.

    Args:
        cls: A registered name (case-insensitive) or an ``eqx.Module`` subclass.

    Returns:
        The corresponding model class.

    Raises:
        ValueError: If ``cls`` is a string not present in the registry.
    """
    if not isinstance(cls, str):
        return cls

    cls_lower = cls.lower()

    # Experimental models are lazy-loaded to avoid importing heavy optional
    # dependencies (TensorFlow, SentencePiece) at module initialisation time.
    if cls_lower == "experimental.textencoder":
        from equimo.experimental.text import TextEncoder

        return TextEncoder

    if cls_lower not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class: {cls!r}. "
            f"Available: {sorted(_MODEL_REGISTRY)}. "
            "Use register_model() to add custom models."
        )
    return _MODEL_REGISTRY[cls_lower]
