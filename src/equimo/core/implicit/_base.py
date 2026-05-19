"""Abstract base classes for DEQ components.

A DEQ cell computes the function ``f(z, x)`` whose fixed point ``z*`` satisfies
``z* = f(z*, x)``. We split the cell into three orthogonal concerns:

* :class:`AbstractInjector` — **how** input ``x`` enters the recurrent state ``z``.
  Injectors may precompute static features of ``x`` once per forward via their
  ``prepare`` method (e.g. a 1x1 projection), so the precomputation is not
  repeated at every fixed-point iteration.
* :class:`AbstractStabilizer` — how the cell output is post-processed to keep
  ``f`` well-behaved (near-contractive). Typical stabilizers project onto a
  normalized manifold (MDEQ-style) or linearly damp between the entry state
  and the stack output.
* :class:`AbstractStackStrategy` — **how** the inner block stack is composed,
  and **where** injection happens in that stack. The strategy is responsible
  for iterating over the blocks; each block is free to keep its own internal
  residual / normalization pattern intact.

The resulting cell is explicit:

.. code-block:: text

    f(z, x):
        h = strategy(blocks, z, x_ctx, injector)    # one or more injections
        return stabilizer(z_in=z, z_out=h, x_ctx)   # contractivity enforcement
"""

from abc import abstractmethod
from typing import Any, TypeAlias

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

# Arbitrary PyTree produced by an injector/stabilizer's ``prepare`` method.
# Kept deliberately permissive so concrete components can cache whatever they
# need (projected tensors, (scale, shift) pairs, etc.).
InputContext: TypeAlias = Any


class AbstractInjector(eqx.Module):
    """Injects input ``x`` into state ``z``.

    Subclasses implement ``__call__(z, x_ctx)`` which must return an array with
    the same shape as ``z``. Override ``prepare`` to cache static transforms of
    ``x`` (projections, embeddings, …) across fixed-point iterations.
    """

    def prepare(self, x: jax.Array, key: PRNGKeyArray) -> InputContext:
        """Precompute static transforms of ``x``. Returns a PyTree passed to ``__call__``."""
        return x

    @abstractmethod
    def __call__(
        self,
        z: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        """Return ``z`` with input injected."""
        ...


class AbstractStabilizer(eqx.Module):
    """Stabilizes the output of ``f(z, x)`` to encourage contractivity.

    Called at the exit of ``f`` with both the original entry state ``z_in``
    (what the solver handed to ``f``) and the stack output ``z_out``. Returns
    the next iterate that will be fed back into the solver.

    Typical choices: identity, norm-projection (``GroupNorm(z_out)``),
    damping (``(1-α) z_in + α z_out``), or their combination.
    """

    def prepare(self, x: jax.Array, key: PRNGKeyArray) -> InputContext:
        return x

    @abstractmethod
    def __call__(
        self,
        z_in: jax.Array,
        z_out: jax.Array,
        x_ctx: InputContext,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array: ...


class AbstractStackStrategy(eqx.Module):
    """Runs the inner block stack of a DEQ cell.

    Encapsulates **where** injection happens (schedule) and any inter-block
    operations (pre-normalization, mid-stack normalization, …). Each block's
    own forward pass is used as-is — the strategy does not bypass, rewrap or
    re-add residuals around a block.
    """

    @abstractmethod
    def __call__(
        self,
        blocks: tuple[eqx.Module, ...],
        z: jax.Array,
        x_ctx: InputContext,
        injector: AbstractInjector,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array: ...
