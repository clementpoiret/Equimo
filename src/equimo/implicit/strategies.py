"""Stack strategies for DEQ cells.

A strategy defines **how** the inner block stack is executed and **where**
input injection happens inside it. Each block is always called with its own
native forward signature — strategies never unwrap or modify a block's
internal residual / normalization.

Available strategies:

* :class:`EntryInjection` — inject **once** at the stack entry. Best default
  for deep stacks of blocks that already normalize internally (ConvNeXt,
  iFormer). Cheap, stable, and the standard choice in MDEQ-style models.
* :class:`PerBlockInjection` — inject **before every block**. More expressive
  and conditioning-rich, at the cost of more compute per iteration. A good
  fit for transformer-like blocks that benefit from persistent input access.
* :class:`ScheduledInjection` — inject at a user-specified set of indices.
  Recovers :class:`EntryInjection` with ``indices=(0,)`` and
  :class:`PerBlockInjection` with ``indices=range(len(blocks))``.
"""

from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._base import AbstractInjector, AbstractStackStrategy, InputContext

_STRATEGY_REGISTRY: dict[str, type[AbstractStackStrategy]] = {}


def register_strategy(
    name: Optional[str] = None,
    force: bool = False,
) -> Callable[[type[AbstractStackStrategy]], type[AbstractStackStrategy]]:
    """Decorator to register a new stack strategy under ``name``."""

    def decorator(cls: type[AbstractStackStrategy]) -> type[AbstractStackStrategy]:
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


def get_strategy(
    module: str | type[AbstractStackStrategy],
) -> type[AbstractStackStrategy]:
    """Resolve a strategy class from its registry name (or pass through)."""
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Got an unknown strategy string: '{module}'. "
            f"Available strategies: {list(_STRATEGY_REGISTRY.keys())}"
        )

    return _STRATEGY_REGISTRY[module_lower]


@register_strategy(name="entry")
class EntryInjection(AbstractStackStrategy):
    """Inject input **once** at the stack entry, then run all blocks as-is.

    Block forward semantics are preserved: each block ``b_i`` receives the
    output of ``b_{i-1}`` and applies its own internal residual connection.
    This is the sensible default for deep stacks of residual blocks, because
    re-injecting ``x`` inside an already-residual stack tends to destabilize
    the fixed-point iteration.

    Pipeline:

    .. code-block:: text

        h = injector(z, x_ctx)
        for b in blocks:
            h = b(h)
        return h
    """

    def __init__(
        self, *, dim: int | None = None, key: PRNGKeyArray | None = None, **kwargs
    ):
        # Accept `dim`/`key` so it constructs under the same factory signature
        # as other components. No parameters to learn here.
        pass

    def __call__(
        self,
        blocks: tuple[eqx.Module, ...],
        z: jax.Array,
        x_ctx: InputContext,
        injector: AbstractInjector,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        k_inj, *k_blocks = jr.split(key, len(blocks) + 1)
        h = injector(z, x_ctx, inference=inference, key=k_inj)
        for blk, kb in zip(blocks, k_blocks):
            h = blk(h, inference=inference, key=kb)
        return h


@register_strategy(name="per_block")
class PerBlockInjection(AbstractStackStrategy):
    """Re-inject the input **before every block**.

    More expressive than :class:`EntryInjection`: useful when the block stack
    is shallow, when blocks lack internal pre-normalization, or when you
    need persistent conditioning on ``x`` at every depth. Costs one extra
    injector call per block per fixed-point iteration.

    Pipeline:

    .. code-block:: text

        h = z
        for b in blocks:
            h = b(injector(h, x_ctx))
        return h
    """

    def __init__(
        self, *, dim: int | None = None, key: PRNGKeyArray | None = None, **kwargs
    ):
        pass

    def __call__(
        self,
        blocks: tuple[eqx.Module, ...],
        z: jax.Array,
        x_ctx: InputContext,
        injector: AbstractInjector,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        keys = jr.split(key, 2 * len(blocks))
        h = z
        for i, blk in enumerate(blocks):
            k_inj, k_blk = keys[2 * i], keys[2 * i + 1]
            h = injector(h, x_ctx, inference=inference, key=k_inj)
            h = blk(h, inference=inference, key=k_blk)
        return h


@register_strategy(name="scheduled")
class ScheduledInjection(AbstractStackStrategy):
    """Inject at user-specified indices in ``[0, len(blocks)]``.

    Index ``i`` means "inject *before* block ``i``"; index ``len(blocks)``
    means "inject after the last block". Useful for experiments like injecting
    at entry and at mid-depth only (e.g. ``indices=(0, 4)``). Indices out of
    range are silently clamped into the valid range at call time.
    """

    indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        indices: Sequence[int] = (0,),
        dim: int | None = None,
        key: PRNGKeyArray | None = None,
        **kwargs,
    ):
        self.indices = tuple(sorted(set(int(i) for i in indices)))
        if not self.indices:
            raise ValueError("`indices` must contain at least one injection point.")

    def __call__(
        self,
        blocks: tuple[eqx.Module, ...],
        z: jax.Array,
        x_ctx: InputContext,
        injector: AbstractInjector,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        n = len(blocks)
        valid_idx = tuple(i for i in self.indices if 0 <= i <= n)
        if not valid_idx:
            # If nothing landed in-range, treat it as an entry injection.
            valid_idx = (0,)

        n_inj = len(valid_idx)
        keys = jr.split(key, n_inj + n)
        inj_keys, blk_keys = keys[:n_inj], keys[n_inj:]

        inj_map: dict[int, PRNGKeyArray] = dict(zip(valid_idx, inj_keys))

        h = z
        for i in range(n):
            if i in inj_map:
                h = injector(h, x_ctx, inference=inference, key=inj_map[i])
            h = blocks[i](h, inference=inference, key=blk_keys[i])
        # Final (post-stack) injection point.
        if n in inj_map:
            h = injector(h, x_ctx, inference=inference, key=inj_map[n])
        return h
