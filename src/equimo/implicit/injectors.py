"""Input injectors for DEQ cells.

Every injector combines the current recurrent state ``z`` with (features of)
the input ``x``. The ``prepare(x)`` hook is called **once per forward pass**
of the DEQ block, so static transforms of ``x`` (projections, embeddings) are
computed only once and reused across all fixed-point iterations.

Available injectors:

* :class:`Add` — shape-matched addition, no parameters. Simplest default.
* :class:`ProjAdd` — 1x1 projection + addition, handles channel mismatch.
* :class:`PreNormAdd` — pre-normalize ``z`` then add a projected ``x``.
  Recommended for deep ConvNeXt / iFormer DEQ stacks: bounds ``z`` at the
  entry without touching each block's internal normalization.
* :class:`Gated` — learned convex mix ``(1−g)·z + g·x`` with per-channel
  sigmoid gate. Caps the forcing term's contribution without an explicit
  normalization; with ``init_gate=0.5`` the cell starts as a damped Picard
  step at init.
* :class:`FiLM` — feature-wise linear modulation. Most expressive, heaviest.
"""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from equimo.layers.norm import LayerNorm2d

from ._base import AbstractInjector, InputContext

_INJECTOR_REGISTRY: dict[str, type[AbstractInjector]] = {}


def register_injector(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[AbstractInjector]], type[AbstractInjector]]:
    """Decorator to register a new injector class under ``name``."""

    def decorator(cls: type[AbstractInjector]) -> type[AbstractInjector]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _INJECTOR_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_INJECTOR_REGISTRY[registry_name]}."
            )

        _INJECTOR_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_injector(module: str | type[AbstractInjector]) -> type[AbstractInjector]:
    """Resolve an injector class from its registry name (or pass through if already a class)."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _INJECTOR_REGISTRY:
        raise ValueError(
            f"Got an unknown injector string: '{module}'. "
            f"Available injectors: {list(_INJECTOR_REGISTRY.keys())}"
        )

    return _INJECTOR_REGISTRY[module_lower]


def _logit(p: float, eps: float = 1e-4) -> float:
    """Numerically safe ``log(p / (1 - p))`` for ``p ∈ (0, 1)``."""
    p = min(max(p, eps), 1.0 - eps)
    return float(jnp.log(p / (1.0 - p)))


@register_injector(name="add")
class Add(AbstractInjector):
    """``z' = z + x``. No learnable parameters. Requires matching shapes."""

    def __init__(self, **kwargs):
        # Accept and discard dim/key for uniform construction.
        pass

    def __call__(
        self,
        z: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        raw_x = x_ctx[0] if isinstance(x_ctx, tuple) else x_ctx
        return z + raw_x


@register_injector(name="proj_add")
class ProjAdd(AbstractInjector):
    """``z' = z + W_x · x`` via a 1x1 convolution.

    The projection ``W_x`` is applied once per forward pass in ``prepare``,
    yielding ``x_ctx = proj(x)`` which is then broadcast-added to ``z`` at every
    fixed-point iteration.
    """

    proj: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        *,
        in_channels: int | None = None,
        use_bias: bool = True,
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.proj = eqx.nn.Conv2d(
            in_channels=in_channels if in_channels is not None else dim,
            out_channels=dim,
            kernel_size=1,
            use_bias=use_bias,
            key=key,
        )

    def prepare(self, x: jax.Array, key: PRNGKeyArray) -> jax.Array:
        return self.proj(x)

    def __call__(
        self,
        z: jax.Array,
        x_ctx: jax.Array,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        return z + x_ctx


@register_injector(name="prenorm_add")
class PreNormAdd(AbstractInjector):
    """``z' = Norm(z) + W_x · x`` — recommended for deep DEQ stacks.

    * ``Norm(z)`` keeps the *entry* of the block stack bounded across
      fixed-point iterations, preventing the Lipschitz constant of ``f`` from
      drifting during training.
    * ``W_x · x`` is precomputed once via :meth:`prepare`.
    * Inner blocks keep their own internal normalization untouched, so this
      combines cleanly with ConvNeXt / iFormer blocks that are already
      pre-normalized internally.
    """

    norm: eqx.Module
    proj: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        *,
        in_channels: int | None = None,
        use_bias: bool = True,
        norm_factory: Callable[[int], eqx.Module] = lambda d: LayerNorm2d(d, eps=1e-6),
        key: PRNGKeyArray,
        **kwargs,
    ):
        self.norm = norm_factory(dim)
        self.proj = eqx.nn.Conv2d(
            in_channels=in_channels if in_channels is not None else dim,
            out_channels=dim,
            kernel_size=1,
            use_bias=use_bias,
            key=key,
        )

    def prepare(self, x: jax.Array, key: PRNGKeyArray) -> jax.Array:
        return self.proj(x)

    def __call__(
        self,
        z: jax.Array,
        x_ctx: jax.Array,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        return self.norm(z) + x_ctx


@register_injector(name="gated")
class Gated(AbstractInjector):
    """Learned convex mix: ``z' = (1 − g) · z + g · x`` with ``g = σ(γ)``.

    Keeps the forcing term bounded via a per-channel sigmoid gate. The gate
    parameter ``γ ∈ ℝ^C`` is stored in logit space and initialized so that
    ``g ≈ init_gate`` at init.

    Init-gate choices:

    * ``init_gate = 0.5`` (default) — at init the cell behaves as
      ``z_{k+1} ≈ 0.5 z_k + 0.5 x``, an implicit damped-Picard step with
      contraction rate ≤ 0.5. Most stable for fresh runs.
    * ``init_gate = 1.0`` — equivalent to plain :class:`Add` at init; the
      gate can *only* reduce the forcing term from there.

    Unlike :class:`PreNormAdd`, this injector does **not** normalize ``z`` —
    it caps the forcing term's *relative magnitude* via the gate instead.
    Requires ``x`` and ``z`` to share a shape.
    """

    gamma: jax.Array  # logit-space gate of shape (dim, 1, 1)

    def __init__(
        self,
        dim: int,
        *,
        init_gate: float = 0.5,
        key: PRNGKeyArray | None = None,
        **kwargs,
    ):
        logit = _logit(init_gate)
        self.gamma = jnp.full((dim, 1, 1), logit, dtype=jnp.float32)

    def __call__(
        self,
        z: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        raw_x = x_ctx[0] if isinstance(x_ctx, tuple) else x_ctx
        gate = jax.nn.sigmoid(self.gamma)
        return (1.0 - gate) * z + gate * raw_x


@register_injector(name="film")
class FiLM(AbstractInjector):
    """Feature-wise linear modulation: ``z' = (1 + γ(x)) · Norm(z) + β(x)``.

    Heavier than :class:`PreNormAdd` but much more expressive: useful when the
    conditioning signal ``x`` needs to modulate the *dynamics* of ``f``, not just
    its fixed point. ``γ`` is initialized so the residual branch starts near
    zero (``γ ≈ 0``), making ``f`` near-contractive at init.
    """

    norm: eqx.Module
    gamma_proj: eqx.nn.Conv2d
    beta_proj: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        *,
        in_channels: int | None = None,
        norm_factory: Callable[[int], eqx.Module] = lambda d: LayerNorm2d(d, eps=1e-6),
        key: PRNGKeyArray,
        **kwargs,
    ):
        k_g, k_b = jr.split(key)
        self.norm = norm_factory(dim)
        ic = in_channels if in_channels is not None else dim
        self.gamma_proj = eqx.nn.Conv2d(ic, dim, kernel_size=1, key=k_g)
        self.beta_proj = eqx.nn.Conv2d(ic, dim, kernel_size=1, key=k_b)
        # Zero-init gamma so (1 + gamma) ≈ 1 at init and the modulation is
        # effectively (Norm(z) + beta(x)) — same starting behavior as PreNormAdd.
        zero_w = self.gamma_proj.weight * 0
        self.gamma_proj = eqx.tree_at(lambda m: m.weight, self.gamma_proj, zero_w)
        if self.gamma_proj.bias is not None:
            zero_b = self.gamma_proj.bias * 0
            self.gamma_proj = eqx.tree_at(lambda m: m.bias, self.gamma_proj, zero_b)

    def prepare(self, x: jax.Array, key: PRNGKeyArray) -> tuple[jax.Array, jax.Array]:
        return self.gamma_proj(x), self.beta_proj(x)

    def __call__(
        self,
        z: jax.Array,
        x_ctx: tuple[jax.Array, jax.Array],
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        gamma, beta = x_ctx
        return (1.0 + gamma) * self.norm(z) + beta
