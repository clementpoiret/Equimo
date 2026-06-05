import re
from typing import Callable, Optional

import equinox as eqx

# Identifier must only contain alphanumerics, hyphens, and underscores.
# This prevents path traversal when identifiers are embedded in local file paths.
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
_MODEL_REGISTRY: dict[str, dict[str | None, type[eqx.Module]]] = {}


def _normalise_modality(modality: str | None) -> str | None:
    return modality.lower() if modality is not None else None


def register_model(
    name: Optional[str] = None,
    force: bool = False,
    modality: str | None = None,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to register a model class under a serialisable string key."""
    registry_modality = _normalise_modality(modality)

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()
        entries = _MODEL_REGISTRY.setdefault(registry_name, {})
        if registry_modality in entries and not force:
            raise ValueError(
                f"Cannot register '{registry_name}' for modality "
                f"{registry_modality!r}. It is already registered to "
                f"{entries[registry_modality]}."
            )

        entries[registry_modality] = cls
        return cls

    return decorator


def get_model_cls(
    cls: str | type[eqx.Module],
    modality: str | None = None,
) -> type[eqx.Module]:
    """Resolve a model class from a registered key or pass through a class."""
    if not isinstance(cls, str):
        return cls

    cls_lower = cls.lower()
    _import_builtin_models(modality)
    if cls_lower not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class: {cls!r}. "
            f"Available: {sorted(_MODEL_REGISTRY)}. "
            "Use register_model() to add custom models."
        )

    entries = _MODEL_REGISTRY[cls_lower]
    if modality is not None:
        registry_modality = _normalise_modality(modality)
        if registry_modality not in entries:
            raise ValueError(
                f"Unknown modality {modality!r} for model class {cls!r}. "
                f"Available modalities: {_format_modalities(entries)}."
            )
        return entries[registry_modality]

    if len(entries) == 1:
        return next(iter(entries.values()))

    raise ValueError(
        f"Ambiguous model class {cls!r}; available modalities: "
        f"{_format_modalities(entries)}. Pass `modality=` to disambiguate."
    )


def _format_modalities(entries: dict[str | None, type[eqx.Module]]) -> list[str]:
    values = sorted(m for m in entries if m is not None)
    if None in entries:
        return ["<unscoped>", *values]
    return values


def _import_builtin_models(modality: str | None) -> None:
    registry_modality = _normalise_modality(modality)
    if registry_modality in (None, "vision"):
        import equimo.vision.models  # noqa: F401
    if registry_modality in (None, "language"):
        import equimo.language.models  # noqa: F401
    if registry_modality in (None, "audio"):
        import equimo.audio.models  # noqa: F401
    if registry_modality in (None, "tabular"):
        import equimo.tabular.models  # noqa: F401
