import banax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import repeat
from jaxtyping import PRNGKeyArray

from ._base import AbstractFuser, AbstractLayerApply, AbstractUpdater, InputContext
from .strategies import StandardLayerApply


def _init_z0(
    x: jax.Array,
    z0: jax.Array | None,
    mode: str = "zeros",
    key: jax.Array = jr.PRNGKey(42),
) -> jax.Array:
    """Initializes state z0. Defaults to zeros matching x shape."""
    if z0 is not None:
        if z0.ndim == 1:
            assert x.ndim == 3, "The method currently only support 3D inputs (CHW)"

            c, h, w = x.shape

            assert (_z0s := z0.shape[0]) == c, (
                f"Shape mismatch between z0 and x along axis 0. Got {_z0s} and {c}."
            )
            z0 = repeat(z0, "c -> c h w", h=h, w=w)

        assert z0.shape == x.shape, f"Shape mismatch: z0 {z0.shape} vs x {x.shape}"
        return z0

    if mode == "zeros":
        return jnp.zeros_like(x)
    elif mode == "ones":
        return jnp.ones_like(x)
    elif mode == "random":
        return jr.normal(key, x.shape, x.dtype) * 0.01
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")


class ImplicitStep(eqx.Module):
    """
    Represents the function f(z, x) in the fixed point equation z = f(z, x).
    Handles the sequence of layers, fusion, and updates.
    """

    layers: list[eqx.Module]
    fuser: AbstractFuser
    global_updater: AbstractUpdater
    internal_updater: AbstractUpdater | None
    layer_strategy: AbstractLayerApply

    def __init__(
        self,
        channels: int,
        depth: int,
        module: type[eqx.Module],
        module_kwargs: dict,
        fuser: AbstractFuser,
        global_updater: AbstractUpdater,
        internal_updater: AbstractUpdater | None = None,
        layer_strategy: AbstractLayerApply = StandardLayerApply(),
        *,
        key: PRNGKeyArray,
    ):
        self.fuser = fuser
        self.global_updater = global_updater
        self.internal_updater = internal_updater
        self.layer_strategy = layer_strategy

        keys = jr.split(key, depth)
        self.layers = [module(**module_kwargs, key=k) for k in keys]

    def prepare_input(
        self, x: jax.Array, key: PRNGKeyArray
    ) -> tuple[InputContext, InputContext]:
        """
        Aggregates precomputations from fuser and updaters.
        Returns: (fuser_context, updater_context)
        """
        k_f, k_u = jr.split(key)
        x_fuser = self.fuser.prepare_input(x, k_f)
        x_updater = self.global_updater.prepare_input(x, k_u)
        return (x_fuser, x_updater)

    def __call__(
        self,
        z: jax.Array,
        x: tuple[InputContext, InputContext],
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        """
        Single step of the DEQ.
        x is a tuple containing precomputed data for fuser and updater.
        """
        x_fuser, x_updater = x

        key_global, key_iupd, *key_layers = jr.split(key, len(self.layers) + 2)

        z_curr = z
        x_in = (
            self.internal_updater.prepare_input(x_fuser, key_iupd)
            if self.internal_updater is not None
            else x_fuser
        )
        for i, layer in enumerate(self.layers):
            # Note: internal updater usually reuses global preparation or just raw x.
            # For simplicity here, we pass x_fuser to layer strategy, assuming
            # internal updater (if continuous flow) can handle it or we strictly use standard strategy.
            z_curr = self.layer_strategy(
                layer=layer,
                z=z_curr,
                x=x_in,
                fuser=self.fuser,
                updater=self.internal_updater,
                inference=inference,
                key=key_layers[i],
            )

        return self.global_updater(
            z, z_curr, x_updater, inference=inference, key=key_global
        )


class DEQBlock(eqx.Module):
    z0_c: jax.Array | None
    function: ImplicitStep
    solver: banax.Solver
    adjoint: banax.Adjoint
    tol: float
    max_steps: int

    def __init__(
        self,
        channels: int,
        depth: int,
        module: type[eqx.Module],
        module_kwargs: dict,
        fuser: AbstractFuser,
        global_updater: AbstractUpdater,
        internal_updater: AbstractUpdater | None,
        layer_strategy: AbstractLayerApply,
        tol: float,
        max_steps: int,
        key: PRNGKeyArray,
        learn_z0: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.z0_c = jnp.zeros(channels, dtype=dtype) if learn_z0 else None
        self.function = ImplicitStep(
            channels=channels,
            depth=depth,
            module=module,
            module_kwargs=module_kwargs,
            fuser=fuser,
            global_updater=global_updater,
            internal_updater=internal_updater,
            layer_strategy=layer_strategy,
            key=key,
        )
        self.solver = banax.solver.Picard(atol=0.0, rtol=tol, max_steps=max_steps)
        self.adjoint = banax.adjoint.NeumannPhantom(
            solver=self.solver, b_solver=banax.solver.Relaxed(damp=0.5, max_steps=5)
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
        key_prep, key_solve, key_reg, key_mix, key_init, key_depth = jr.split(key, 6)

        # precompute static projections
        x_context = self.function.prepare_input(x, key_prep)

        # Topological Support Expansion (Mixed Initialization)
        if not inference:
            z0_zeros = jnp.zeros_like(x) if z0 is None else z0
            z0_noise = jr.normal(key_init, x.shape, x.dtype) * 0.1

            mask = jr.bernoulli(key_mix, p=0.5)
            z0_val = jax.lax.select(mask, z0_noise, z0_zeros)
        else:
            z0_val = jnp.zeros_like(x) if z0 is None else z0

        z0_sg = jax.lax.stop_gradient(z0_val)

        f = eqx.Partial(self.function, inference=inference, key=key_solve)

        sol = self.adjoint((f, (x_context,)), z0_sg)
        z_star = sol.value
        trj = sol.trace
        depth = sol.stats.steps
        error = sol.stats.rel_err

        aux = {
            "key": key_solve,
            "x_context": x_context,  # passing for later regs in loss
            "z0": z0,
            "z_star": z_star,
            "trajectory": trj,
            "depth": depth,
            "error": error,
        }

        return z_star, aux
