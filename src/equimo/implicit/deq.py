"""DEQ cell and block.

:class:`DEQCell` is the pure function ``f(z, x)`` whose fixed point the solver
finds. It composes three orthogonal pieces:

* an :class:`~._base.AbstractInjector` that injects input ``x`` into state ``z``,
* an :class:`~._base.AbstractStackStrategy` that runs the inner block stack
  (and schedules injection points),
* an :class:`~._base.AbstractStabilizer` that post-processes the stack output
  to keep ``f`` near-contractive.

:class:`DEQBlock` wraps a :class:`DEQCell` with a fixed-point solver and an
implicit-differentiation adjoint (both from ``banax``), exposing the usual
``z_star, aux = block(x)`` API.
"""

import banax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import repeat
from jaxtyping import PRNGKeyArray

from ._base import (
    AbstractInjector,
    AbstractStabilizer,
    AbstractStackStrategy,
    InputContext,
)
from .strategies import EntryInjection


def _init_z0(
    x: jax.Array,
    z0: jax.Array | None,
    mode: str = "zeros",
    inference: bool = False,
    key: PRNGKeyArray = jr.PRNGKey(42),
) -> jax.Array:
    """Initialize ``z0`` for the fixed-point solver.

    Modes:
        * ``"zeros"`` / ``"ones"`` / ``"random"``: trivial initializations.
        * ``"mixed"``: *Topological Support Expansion* — during training,
          randomly pick between the base init (``z0`` if provided, else zeros)
          and small Gaussian noise with Bernoulli p=0.5. At inference it
          falls back to the base init deterministically.
    """
    if z0 is not None:
        if z0.ndim == 1:
            assert x.ndim == 3, "The method currently only supports 3D inputs (CHW)."

            c, h, w = x.shape

            assert (_z0s := z0.shape[0]) == c, (
                f"Shape mismatch between z0 and x along axis 0. Got {_z0s} and {c}."
            )
            z0 = repeat(z0, "c -> c h w", h=h, w=w)

        assert z0.shape == x.shape, f"Shape mismatch: z0 {z0.shape} vs x {x.shape}"

    if mode == "mixed":
        z0_base = jnp.zeros_like(x) if z0 is None else z0
        if inference:
            return z0_base
        key_mix, key_noise = jr.split(key)
        z0_noise = jr.normal(key_noise, x.shape, x.dtype) * 0.1
        mask = jr.bernoulli(key_mix, p=0.5)
        return jax.lax.select(mask, z0_noise, z0_base)

    if z0 is not None:
        return z0

    if mode == "zeros":
        return jnp.zeros_like(x)
    elif mode == "ones":
        return jnp.ones_like(x)
    elif mode == "random":
        return jr.normal(key, x.shape, x.dtype) * 0.01
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")


class DEQCell(eqx.Module):
    """The function ``f(z, x)`` whose fixed point is the DEQ output.

    Explicit pipeline, per call:

    .. code-block:: text

        h   = strategy(blocks, z, x_inj, injector)   # where/how to inject x
        z'  = stabilizer(z_in=z, z_out=h, x_stab)    # contractivity enforcement

    The ``prepare`` method is called once per outer forward of :class:`DEQBlock`
    and caches static transforms of ``x`` for both the injector and the
    stabilizer, so they are not recomputed at every fixed-point iteration.
    """

    blocks: tuple[eqx.Module, ...]
    injector: AbstractInjector
    stabilizer: AbstractStabilizer
    strategy: AbstractStackStrategy

    def __init__(
        self,
        *,
        channels: int,  # kept for API compatibility; individual blocks know their own dim
        depth: int,
        module: type[eqx.Module],
        module_kwargs: dict,
        injector: AbstractInjector,
        stabilizer: AbstractStabilizer,
        strategy: AbstractStackStrategy = EntryInjection(),
        key: PRNGKeyArray,
    ):
        del channels  # unused here; consumed by submodules via module_kwargs

        self.injector = injector
        self.stabilizer = stabilizer
        self.strategy = strategy

        keys = jr.split(key, depth)
        self.blocks = tuple(module(**module_kwargs, key=k) for k in keys)

    def prepare(
        self, x: jax.Array, key: PRNGKeyArray
    ) -> tuple[InputContext, InputContext]:
        """Precompute (injector_ctx, stabilizer_ctx) from ``x``.

        Called **once per outer forward**, before the fixed-point solver loop.
        """
        k_i, k_s = jr.split(key)
        return (
            self.injector.prepare(x, k_i),
            self.stabilizer.prepare(x, k_s),
        )

    def __call__(
        self,
        z: jax.Array,
        x: tuple[InputContext, InputContext],
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        inj_ctx, stab_ctx = x
        k_stack, k_stab = jr.split(key)

        h = self.strategy(
            blocks=self.blocks,
            z=z,
            x_ctx=inj_ctx,
            injector=self.injector,
            inference=inference,
            key=k_stack,
        )
        return self.stabilizer(
            z_in=z, z_out=h, x_ctx=stab_ctx, inference=inference, key=k_stab
        )


class DEQBlock(eqx.Module):
    """Fixed-point wrapper around a :class:`DEQCell`.

    Owns the solver and adjoint. The adjoint uses a Neumann-phantom gradient
    with a relaxed Picard backward solve, which is cheap and works well for
    ConvNeXt-style stacks without requiring full Jacobian inversion.
    """

    cell: DEQCell
    solver: banax.Solver
    adjoint: banax.Adjoint
    tol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        channels: int,
        depth: int,
        module: type[eqx.Module],
        module_kwargs: dict,
        injector: AbstractInjector,
        stabilizer: AbstractStabilizer,
        strategy: AbstractStackStrategy,
        tol: float,
        max_steps: int,
        key: PRNGKeyArray,
    ):
        self.cell = DEQCell(
            channels=channels,
            depth=depth,
            module=module,
            module_kwargs=module_kwargs,
            injector=injector,
            stabilizer=stabilizer,
            strategy=strategy,
            key=key,
        )
        self.solver = banax.solver.Picard(atol=0.0, rtol=tol, max_steps=max_steps)
        self.adjoint = banax.adjoint.NeumannPhantom(
            solver=self.solver,
            b_solver=banax.solver.Relaxed(damp=0.5, max_steps=5),
        )
        self.tol = tol
        self.max_steps = max_steps

    def __call__(
        self,
        x: jax.Array,
        z0: jax.Array | None = None,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ):
        key_prep, key_solve, key_init = jr.split(key, 3)

        # Static precomputations on x (shared across all Picard iterations).
        x_context = self.cell.prepare(x, key_prep)

        z0_val = _init_z0(x, z0, mode="mixed", inference=inference, key=key_init)
        z0_sg = jax.lax.stop_gradient(z0_val)

        # The same key is reused across fixed-point iterations on purpose: it
        # freezes any stochastic ops inside the cell (DropPath, Dropout) so
        # that f is deterministic per-example, which is required for the
        # fixed-point iteration to converge.
        f = eqx.Partial(self.cell, inference=inference, key=key_solve)

        sol = self.adjoint((f, (x_context,)), z0_sg)
        z_star = sol.value
        trj = sol.trace
        depth = sol.stats.steps
        error = sol.stats.rel_err

        aux = {
            "key": key_solve,
            "x_context": x_context,  # passed through for downstream regularizers
            "z0": z0,
            "z_star": z_star,
            "trajectory": trj,
            "depth": depth,
            "error": error,
        }

        return z_star, aux
