# ty: ignore[invalid-assignment]
__all__ = [
    "SpectrogramPatchEmbedding",
    "get_patch",
    "register_patch",
]

from typing import Callable, Optional, Tuple

import equinox as eqx
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.utils import make_2tuple

_PATCH_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_patch(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Register an audio patch embedding module."""

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _PATCH_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_PATCH_REGISTRY[registry_name]}."
            )

        _PATCH_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_patch(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get an audio patch embedding module class from its registered name."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _PATCH_REGISTRY:
        raise ValueError(
            f"Got an unknown audio patch module string: '{module}'. "
            f"Available modules: {list(_PATCH_REGISTRY.keys())}"
        )

    return _PATCH_REGISTRY[module_lower]


@register_patch()
class SpectrogramPatchEmbedding(eqx.Module):
    """Patch embedding for single-channel log-mel spectrograms.

    Inputs follow the AST convention ``(time, frequency)`` and are internally
    projected as ``(1, frequency, time)`` so ``fstride`` and ``tstride`` map to
    the same axes as the AST reference implementation.
    """

    patch_size: Tuple[int, int] = eqx.field(static=True)
    stride: Tuple[int, int] = eqx.field(static=True)
    img_size: Tuple[int, int] = eqx.field(static=True)
    grid_size: Tuple[int, int] = eqx.field(static=True)
    num_patches: int = eqx.field(static=True)

    proj: eqx.nn.Conv

    def __init__(
        self,
        dim: int,
        patch_size: int | Tuple[int, int],
        *,
        input_fdim: int,
        input_tdim: int,
        fstride: int,
        tstride: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = make_2tuple(patch_size)
        self.stride = (fstride, tstride)
        self.img_size = (input_fdim, input_tdim)

        f_dim = (input_fdim - self.patch_size[0]) // fstride + 1
        t_dim = (input_tdim - self.patch_size[1]) // tstride + 1
        if f_dim <= 0 or t_dim <= 0:
            raise ValueError(
                "Patch size must fit within the configured spectrogram dimensions."
            )

        self.grid_size = (f_dim, t_dim)
        self.num_patches = f_dim * t_dim
        self.proj = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=1,
            out_channels=dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            key=key,
        )

    def __call__(
        self, x: Float[Array, "time frequency"]
    ) -> Float[Array, "num_patches dim"]:
        T, F = x.shape
        if T != self.img_size[1]:
            raise AssertionError(
                f"Input time dimension ({T}) doesn't match model ({self.img_size[1]})"
            )
        if F != self.img_size[0]:
            raise AssertionError(
                f"Input frequency dimension ({F}) doesn't match model ({self.img_size[0]})"
            )

        x = rearrange(x, "t f -> 1 f t")
        x = self.proj(x)
        return rearrange(x, "c f t -> (f t) c")
