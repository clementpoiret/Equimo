import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.core.layers.ffn import Mlp


def _call_mlp(
    mlp: Mlp,
    x: Float[Array, "... dim"],
    *,
    key: PRNGKeyArray | None = None,
    inference: bool | None = None,
) -> Float[Array, "... out_dim"]:
    in_shape = x.shape[:-1]
    x_flat = x.reshape((-1, x.shape[-1]))
    if key is None:
        key = jr.PRNGKey(0)
    out = mlp(x_flat, key=key, inference=inference)
    return out.reshape((*in_shape, out.shape[-1]))
