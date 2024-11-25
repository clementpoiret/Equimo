# Based on the original torch pscan impl. from radarFudan:
# https://github.com/radarFudan/mamba-minimal-jax/blob/b76334404f7f1d87e47ffc1158b1bd151098d1c2/model.py
# Mamba2-related code from https://github.com/walln/scratch/blob/ab0b6b891830375b7aa64c8e46e77783b843f5ca/src/scratch/language_modeling/mamba/mamba.py#L462
import jax
import jax.numpy as jnp
from einops import einsum, rearrange, repeat
from jax import lax


# INFO: This is simply the implementation from the jimmy project (working)
#       but with batch size removed because Equinox relies on `jax.vmap`.
def selective_scan(
    u: jnp.ndarray,
    delta: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray,
    delta_bias: jnp.ndarray | None = None,
    delta_softplus: bool = False,
):
    """Performs the selective scan algorithm as described in the Mamba paper.

    This function implements the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    where B and C (and the step size delta, which is used for discretization)
    are dependent on the input x(t).

    Args:
        u (jnp.ndarray): Input tensor of shape (d, l).
        delta (jnp.ndarray): Step size tensor of shape (d, l).
        A (jnp.ndarray): State transition matrix of shape (d, n).
        B (jnp.ndarray): Input matrix of shape (n, l).
        C (jnp.ndarray): Output matrix of shape (n, l).
        D (jnp.ndarray): Direct feedthrough matrix of shape (d,).
        delta_bias (jnp.ndarray | None, optional): Bias for delta. Defaults to None.
        delta_softplus (bool, optional): Whether to apply softplus to delta. Defaults to False.

    Returns:
        jnp.ndarray: Output tensor of shape (d, l).

    References:
        [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
        [2] The Annotated S4: run_SSM(A, B, C, u)

    Notes:
        - l: sequence length
        - d: hidden dimension
        - n: latent space dimension

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
    """
    d_in, l = u.shape

    if delta_bias is not None:
        delta = delta + jnp.expand_dims(delta_bias, axis=-1)
    if delta_softplus:
        delta = jax.nn.softplus(delta)

    # Discretize continuous parameters (A, B)
    deltaA = jnp.exp(einsum(delta, A, "d l, d n -> l d n"))
    deltaB_u = einsum(delta, B, u, "d l, n l, d l -> l d n")

    # Define the scan function
    def scan_fn(carry, x):
        x_prev, _ = carry
        x_next = deltaA[:, x] * x_prev + deltaB_u[:, x]
        y = einsum(x_next, C[:, :, x], "d n, n -> d")
        return (x_next, None), y

    x_init = jnp.zeros((d_in, A.shape[1]))
    carry_init = (x_init, None)
    _, ys = lax.scan(scan_fn, carry_init, jnp.arange(l))
    ys = rearrange(ys, "l d -> d l")

    y = ys + u * jnp.expand_dims(D, axis=-1)

    return y


def segsum(x: jnp.ndarray):
    """Stable segment sum calculation.

    Produces a 1-semiseperable matrix which is equivalent to a scalar SSM.

    Args:
        x (seq_len, n_heads): input tensor

    Returns:
        output tensor of shape (seq_len, n_heads)
    """
    T = x.shape[-1]
    x = repeat(x, "... d -> ... d e", e=T)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), -1)
    x = jnp.where(mask, x, 0)
    x_segsum = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool), 0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def find_closest_divisor(n: int, target: int):
    """Find the closest divisor of n to the target value."""
    divisors = jnp.arange(1, n + 1)
    is_divisor = n % divisors == 0

    # Replace non-divisors with a large value
    large_value = jnp.iinfo(jnp.int32).max
    valid_divisors = jnp.where(is_divisor, divisors, large_value)

    # Find the index of the closest divisor
    closest_index = jnp.argmin(jnp.abs(valid_divisors - target))

    # Return the closest divisor
    return divisors[closest_index]


def ssd(
    x: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    chunk_size: int,
    initial_states: jnp.ndarray | None = None,
):
    """Structured State Space Duality (SSD).

    This function implements the SSD algorithm for computing the SSM states. It
    takes in the input tensor, A, B, and C, and returns the output tensor, y, and
    the updated SSM states. The SSD algorithm is a generalization of the SSM
    algorithm to the case where the SSM states are not scalars. It is a
    structured matrix multiplication that is equivalent to a scalar SSM.

    Args:
        u (jnp.ndarray): Input tensor of shape (l, n, d_head).
        A (jnp.ndarray): State transition matrix of shape (d, l, n).
        B (jnp.ndarray): Input matrix of shape (l, n, d_state).
        C (jnp.ndarray): Output matrix of shape (l, n, d_state).
        chunk_size: matrix partition size.
        initial_states: (1, n, d_state)

    Returns:
        y (jnp.ndarray): Output tensor of shape (l, n, d_head).
        state (jnp.ndarray): Output tensor of shape (l, n, d_state).

    Notes:
        - l: sequence length
        - d: hidden dimension
        - n: number of heads

    Implementation taken from:
        https://github.com/walln/scratch/blob/ab0b6b891830375b7aa64c8e46e77783b843f5ca/src/scratch/language_modeling/mamba/mamba.py#L537
    """
    # adjusted_chunk_size = find_closest_divisor(x.shape[0], chunk_size)
    assert x.shape[0] % chunk_size == 0

    # Rearrange into chunks
    x, A, B, C = (
        rearrange(m, "(c l) ... -> c l ...", l=chunk_size) for m in (x, A, B, C)
    )

    A = rearrange(A, "c l h -> h c l")
    A_cumsum = jnp.cumsum(A, axis=-1)

    # Compute intra-chunk state (diagonal blocks)
    L = jnp.exp(segsum(A))
    Y_diag = jnp.einsum("clhn, cshn, hcls, cshp -> clhp", C, B, L, x)

    # Compute intra-chunk state - the right term of low rank factorization of the
    # off diagonal blocks; B terms
    decay_states = jnp.exp(A_cumsum[:, :, -1:] - A_cumsum)
    states = jnp.einsum("clhn, hcl, clhp -> chpn", B, decay_states, x)

    # Compute the inter-chunk SSM recurrence. Producing the correct SSM states at chunk
    # boundaries. This is the middle term of off diagonal blocks; A terms.
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])

    states = jnp.concat([initial_states, states], axis=1)
    decay_chunk = jnp.exp(segsum(jnp.pad(A_cumsum[:, :, -1], ((0, 0), (0, 0), (1, 0)))))
    new_states = jnp.einsum("hzc, chpn -> zhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]  # TODO: check

    # Compute state and output conversion per chunk
    # the left term of low rank factorization of the off diagonal blocks; C terms
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum("clhn, chpn, hcl -> clhp", C, states, state_decay_out)

    # Add the output of intra-chunk and inter-chunk states
    Y = rearrange(Y_diag + Y_off, "c l h p -> (c l) h p")

    return Y, final_state


def non_causal_linear_attn(
    x: jnp.ndarray,
    dt: jnp.ndarray,
    A: jnp.ndarray,
    B: jnp.ndarray,
    C: jnp.ndarray,
    D: jnp.ndarray,
    n_groups: int = 1,
):
    """Non-causal attention duality from the VSSD paper."""
    l, h, d = x.shape
    d_state = B.shape[1]  # TODO: initially 2, check
    V = rearrange(x, "l h d -> h l d")
    dt = rearrange(dt, "l h -> h l")
    dA = dt[..., None] * jnp.broadcast_to(A[:, None, None], (A.shape[0], l, 1))

    V_scaled = V * dA
    K = jnp.reshape(B, (1, l, d_state))

    if n_groups == 1:
        # get kv via transpose K and V
        KV = jnp.matmul(jnp.swapaxes(K, -2, -1), V_scaled)
        Q = jnp.reshape(C, (1, l, d_state))
        x = jnp.matmul(Q, KV)
        x = x + V * jnp.broadcast_to(D[:, None, None], (D.shape[0], l, 1))
        x = rearrange(x, "h l d -> l h d")
    else:
        if h % n_groups != 0:
            raise ValueError("h % g != 0")
        d_state = d_state // n_groups
        K = jnp.transpose(
            jnp.reshape(K, (1, l, n_groups, d_state)),
            (0, 2, 1, 3),
        )
        V_scaled = jnp.reshape(V_scaled, (h // n_groups, n_groups, l, d))
        Q = jnp.transpose(
            jnp.reshape(C, (1, l, n_groups, d_state)),
            (0, 2, 1, 3),
        )

        KV = jnp.matmul(jnp.swapaxes(K, -2, -1), V_scaled)
        x = jnp.matmul(Q, KV)
        V_skip = jnp.reshape(
            V * jnp.broadcast_to(D[:, None, None], (D.shape[0], l, 1)),
            (h // n_groups, n_groups, l, d),
        )
        x = x + V_skip
        x = jnp.reshape(jnp.transpose(x, (2, 0, 1, 3)), (l, h, d))

    return x
