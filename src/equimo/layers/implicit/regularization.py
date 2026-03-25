from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def hutchinson_jacobian_frobenius(
    f: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    z_star: Float[Array, "..."],
    *,
    f_args: tuple = (),
    f_kwargs: dict = {},
    n_steps: int = 4,
    key: PRNGKeyArray,
    distribution: Literal["rademacher", "sphere"] = "rademacher",
) -> Float[Array, ""]:
    """
    Estimate the re-scaled squared Frobenius norm
      of the jacobian of f at z_star
      using Hutchinson's algorithm,
      i.e., estimate tr(J^T J)/d
      where J := (df/dz)(z_star)
      and d is the dimensionality of z_star.
    f: function z -> f(z).
    z_star: equilibrium point tensor.
    n_steps: number of random vectors used in Hutchinson's algorithm.
    key: PRNGKey.
    distribution: "rademacher" (default, lower variance) or "sphere".
    """
    _f = lambda z: f(z, *f_args, **f_kwargs)

    d = z_star.size
    keys = jr.split(key, n_steps)
    _, vjp_fn = eqx.filter_vjp(_f, z_star)

    def sample_rademacher(k):
        return jr.rademacher(k, z_star.shape, dtype=z_star.dtype)

    def sample_sphere(k):
        # Sample on unit sphere, then scale by sqrt(d) so E[v v^T] = I
        v = jr.normal(k, z_star.shape, dtype=z_star.dtype)
        return v * (jnp.sqrt(d) / jnp.linalg.norm(v))

    if distribution == "rademacher":
        sampler = sample_rademacher
    elif distribution == "sphere":
        sampler = sample_sphere
    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. Use 'rademacher' or 'sphere'."
        )

    eps = jax.vmap(sampler)(keys)
    j_eps = jax.vmap(lambda v: vjp_fn(v)[0])(eps)

    sq_norms = jnp.sum(j_eps**2, axis=range(1, j_eps.ndim))
    return jnp.mean(sq_norms) / d


def denoising(
    f: Callable[[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]],
    z_star: Float[Array, "..."],
    sigma: float = 0.01,
    n_probes: int = 1,
    *,
    f_args: tuple = (),
    f_kwargs: dict = {},
    key: PRNGKeyArray,
) -> Float[Array, "1"]:
    """
    Implements Denoising Regularization for DEQs.

    This term regularizes the Jacobian at the fixed-point by adding noise
    to the equilibrium state and penalizing the network's deviation
    from that state.
    """
    # Sample isotropic Gaussian noise: epsilon ~ N(0, I * sigma^2)
    epsilon = jr.normal(key, (n_probes,) + z_star.shape) * sigma

    z_noisy = z_star + epsilon

    # Calculate f(z* + epsilon)
    # The paper uses ||z* - f(z* + eps, x)||^2
    # This induces a penalty on the Frobenius norm of the Jacobian.
    f_vec = jax.vmap(lambda z: f(z, *f_args, **f_kwargs))
    f_noisy = f_vec(z_noisy)

    diff = z_star - f_noisy

    return jnp.mean(jnp.square(diff)) / (sigma**2)
