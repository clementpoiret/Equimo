"""Output stabilizers for DEQ cells.

A stabilizer is applied at the **exit** of ``f(z, x)`` — it sees both the
entry state ``z_in`` that the solver handed to ``f`` and the raw stack output
``z_out``. Its role is to keep the map ``f`` near-contractive so the fixed-point
solver actually converges.

Common strategies, increasing in strength:

* :class:`Identity` — pass through. Only safe if the block stack itself is
  already strongly contractive (e.g. near-identity ConvNeXt init with
  ``LayerScale(1e-6)`` and a short stack).
* :class:`Damped` — convex combination ``(1-α) z_in + α z_out`` with a
  learnable scalar / per-channel ``α``. Works well when the stack drift is
  small but nonzero.
* :class:`GroupNormProject` — projects ``z_out`` onto the normalized manifold
  (MDEQ-style). Robust default for deep stacks because it hard-caps the norm
  of the iterate.
* :class:`DampedProject` — combines damping with norm projection. Best when
  you want both bounded iterates and a smooth trajectory.
"""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from equimo.utils import nearest_power_of_2_divisor

from ._base import AbstractStabilizer, InputContext

_STABILIZER_REGISTRY: dict[str, type[AbstractStabilizer]] = {}


def register_stabilizer(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[AbstractStabilizer]], type[AbstractStabilizer]]:
    """Decorator to register a new stabilizer class under ``name``."""

    def decorator(cls: type[AbstractStabilizer]) -> type[AbstractStabilizer]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _STABILIZER_REGISTRY and not force:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_STABILIZER_REGISTRY[registry_name]}."
            )

        _STABILIZER_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_stabilizer(module: str | type[AbstractStabilizer]) -> type[AbstractStabilizer]:
    """Resolve a stabilizer class from its registry name (or pass through)."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _STABILIZER_REGISTRY:
        raise ValueError(
            f"Got an unknown stabilizer string: '{module}'. "
            f"Available stabilizers: {list(_STABILIZER_REGISTRY.keys())}"
        )

    return _STABILIZER_REGISTRY[module_lower]


@register_stabilizer(name="identity")
class Identity(AbstractStabilizer):
    """``z_new = z_out``. Pass-through; relies on the stack itself being well-behaved."""

    def __init__(self, **kwargs):
        pass

    def __call__(
        self,
        z_in: jax.Array,
        z_out: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        return z_out


@register_stabilizer(name="projected")
class GroupNormProject(AbstractStabilizer):
    """``z_new = GroupNorm(z_out)``.

    Projects the stack output onto the normalized manifold (MDEQ-style). Caps
    the iterate's norm, which makes ``f`` bounded and typically near-contractive
    for ConvNeXt / iFormer stacks. Recommended default for deep stages.
    """

    norm: eqx.nn.GroupNorm

    def __init__(self, dim: int, *, num_groups: int = 32, eps: float = 1e-6, **kwargs):
        groups = nearest_power_of_2_divisor(dim, num_groups)
        self.norm = eqx.nn.GroupNorm(groups, dim, eps=eps)

    def __call__(
        self,
        z_in: jax.Array,
        z_out: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        return self.norm(z_out)


@register_stabilizer(name="damped")
class Damped(AbstractStabilizer):
    """``z_new = (1 - α) z_in + α z_out``.

    ``α`` can be a fixed scalar, a learnable scalar (``mode="scalar"``), or a
    learnable per-channel vector (``mode="channel"``). All learnable variants
    are initialized to ``init_alpha`` (default 1.0 — full update) and kept in
    ``[0, 1]`` by reparametrization through a sigmoid for stability.
    """

    alpha_param: jax.Array | None  # learnable logit; None when alpha is fixed
    fixed_alpha: float | None = eqx.field(static=True)
    mode: str = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        init_alpha: float = 1.0,
        learnable: bool = True,
        mode: str = "scalar",  # "scalar" or "channel"
        **kwargs,
    ):
        assert 0.0 <= init_alpha <= 1.0, "init_alpha must be in [0, 1]."
        assert mode in ("scalar", "channel")

        self.mode = mode
        if learnable:
            # Inverse-sigmoid of init_alpha gives the starting logit.
            # Clamp to avoid inf when init_alpha is exactly 0 or 1.
            a = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
            logit = float(jnp.log(a / (1.0 - a)))
            shape = (1,) if mode == "scalar" else (dim, 1, 1)
            self.alpha_param = jnp.full(shape, logit, dtype=jnp.float32)
            self.fixed_alpha = None
        else:
            self.alpha_param = None
            self.fixed_alpha = float(init_alpha)

    def _alpha(self) -> jax.Array | float:
        if self.alpha_param is not None:
            return jax.nn.sigmoid(self.alpha_param)
        return self.fixed_alpha

    def __call__(
        self,
        z_in: jax.Array,
        z_out: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        a = self._alpha()
        return (1.0 - a) * z_in + a * z_out


@register_stabilizer(name="damped_projected")
class DampedProject(AbstractStabilizer):
    """``z_new = (1 - α) z_in + α · GroupNorm(z_out)``.

    Combines damping with norm projection. The projection keeps ``z_out``
    bounded, and damping smooths the trajectory across fixed-point iterations.
    """

    norm: eqx.nn.GroupNorm
    alpha_param: jax.Array | None
    fixed_alpha: float | None = eqx.field(static=True)
    mode: str = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        *,
        num_groups: int = 32,
        eps: float = 1e-6,
        init_alpha: float = 1.0,
        learnable: bool = True,
        mode: str = "scalar",
        **kwargs,
    ):
        assert 0.0 <= init_alpha <= 1.0
        assert mode in ("scalar", "channel")

        groups = nearest_power_of_2_divisor(dim, num_groups)
        self.norm = eqx.nn.GroupNorm(groups, dim, eps=eps)
        self.mode = mode
        if learnable:
            a = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
            logit = float(jnp.log(a / (1.0 - a)))
            shape = (1,) if mode == "scalar" else (dim, 1, 1)
            self.alpha_param = jnp.full(shape, logit, dtype=jnp.float32)
            self.fixed_alpha = None
        else:
            self.alpha_param = None
            self.fixed_alpha = float(init_alpha)

    def _alpha(self) -> jax.Array | float:
        if self.alpha_param is not None:
            return jax.nn.sigmoid(self.alpha_param)
        return self.fixed_alpha

    def __call__(
        self,
        z_in: jax.Array,
        z_out: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        a = self._alpha()
        return (1.0 - a) * z_in + a * self.norm(z_out)
