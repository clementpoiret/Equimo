from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def make_2tuple(x):
    """Convert input into a 2-tuple.

    Args:
        x: Input value, either an integer or a 2-tuple

    Returns:
        tuple: If input is integer, returns (x,x). If input is 2-tuple, returns it unchanged.

    Raises:
        AssertionError: If input is tuple but not length 2, or if input is not int or tuple
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


def to_list(obj, n):
    """Convert an object to a list of length n by repeating it or validating existing list.

    Args:
        obj: Input object or list
        n: Desired length of output list

    Returns:
        List of length n containing obj repeated n times if obj is not a list,
        or the original list if it's already length n

    Raises:
        AssertionError: If obj is a list but its length doesn't match n
    """
    if isinstance(obj, list):
        if len(obj) == n:
            return obj
        else:
            raise AssertionError(
                f"obj (list of len {len(obj)}) should have a size of {n}"
            )

    return [obj] * n


def nearest_power_of_2_divisor(dim: int, max: int):
    """Find the largest power of 2 that divides dim, up to a maximum value.

    Args:
        dim: The number to find divisors for
        max: Maximum value to consider (must be a power of 2)

    Returns:
        int: Largest power of 2 that divides dim, not exceeding max

    Example:
        >>> nearest_power_of_2_divisor(24, 32)
        8  # because 8 is the largest power of 2 <= 32 that divides 24
    """
    power = 1
    nearest = 1
    while power <= max:
        if dim % power == 0:
            nearest = power
        power *= 2
    return nearest


@partial(
    jax.jit,
    static_argnames=[
        "pool_type",
        "num_prefix_tokens",
        "reduce_include_prefix",
    ],
)
def pool_sd(
    x: Float[Array, "seqlen dim"],
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    """Pool sequence dimension using various strategies.

    Args:
        x: Input tensor of shape (sequence_length, dimension)
        pool_type: Pooling strategy to use:
            - "token": Use first token (typically CLS token)
            - "avg": Average pooling
            - "max": Max pooling
            - "avgmax": Average of max and mean pooling
            - "": No pooling (return input unchanged)
        num_prefix_tokens: Number of special tokens at start of sequence
        reduce_include_prefix: Whether to include prefix tokens in pooling

    Returns:
        Pooled tensor. If pool_type is "token", returns vector of size dim.
        For other pool types, returns reduced tensor according to the strategy.

    Raises:
        ValueError: If pool_type is not one of the supported values
    """
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[0]  # class token
    else:
        x = x if reduce_include_prefix else x[num_prefix_tokens:]
        match pool_type:
            case "avg":
                x = jnp.mean(x, axis=0)
            case "avgmax":
                x = 0.5 * (jnp.max(x, axis=0) + jnp.mean(x, axis=0))
            case "max":
                x = jnp.max(x, axis=0)
            case _:
                raise ValueError(f"Unknown pool type {pool_type}")

    return x
